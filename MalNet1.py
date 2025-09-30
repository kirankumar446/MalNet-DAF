
"""
malaria_cnn_sam_lstm_tam.py

Full pipeline:
 - Load images from IMAGE_DIR/{Parasitized,Uninfected}
 - Preprocess & split (stratified)
 - Build hybrid model: CNN backbone -> SAM (spatial attention) -> Flatten->reshape->BiLSTM -> Temporal Attention (TAM)
 - Concatenate SAM & TAM vectors -> Dense -> sigmoid
 - Train with callbacks, evaluate (AUC + Youden threshold), save model
 - Optional: extract temporal attention weights for visualization

Update IMAGE_DIR at the top before running.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K, callbacks

# ---------------------------
# User settings - update these
# ---------------------------
IMAGE_DIR = r"E:\\data-kiran\\Recovered Data MSI\\B Tech Projects AITAM\\Batch B9 2023-2024\\images\\cell_images"
# IMAGE_DIR must contain two subfolders: 'Parasitized' and 'Uninfected'
SIZE = 150              # image resize (SIZE x SIZE)
BATCH_SIZE = 32
EPOCHS = 12
RANDOM_STATE = 42

# model hyperparams
BASE_FILTERS = 32
BLOCKS = 4
DROP_RATE = 0.25
LSTM_UNITS = 128
TIMESTEPS = 8
DENSE_UNITS = 128

MODEL_SAVE_PATH = "malaria_cnn_sam_lstm_tam.h5"

# ---------------------------
# Robust loader
# ---------------------------
def load_images_from_folder(folder, label, size=SIZE):
    imgs = []
    labels = []
    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
    for fname in sorted(os.listdir(folder)):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in valid_ext:
            continue
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        img = img.resize((size, size))
        arr = np.asarray(img)
        imgs.append(arr)
        labels.append(label)
    return imgs, labels

def build_dataset(image_dir):
    pos_dir = os.path.join(image_dir, 'Parasitized')
    neg_dir = os.path.join(image_dir, 'Uninfected')
    assert os.path.isdir(pos_dir) and os.path.isdir(neg_dir), "Ensure Parasitized/ and Uninfected/ subfolders exist."

    p_imgs, p_lbls = load_images_from_folder(pos_dir, 0)
    u_imgs, u_lbls = load_images_from_folder(neg_dir, 1)

    X = np.array(p_imgs + u_imgs)
    y = np.array(p_lbls + u_lbls)

    # shuffle then split stratified
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=RANDOM_STATE, stratify=y)
    # normalize floats
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0

    print(f"Loaded dataset: Train samples={len(X_train)}, Test samples={len(X_test)}, Image shape={X_train.shape[1:]}")
    return X_train, X_test, y_train, y_test

# ---------------------------
# Model building utilities
# ---------------------------
def conv_block(x, filters, kernel_size=(3,3), pool=True, dropout=0.0, name=None):
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', name=None if not name else f"{name}_conv1")(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', name=None if not name else f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=None if not name else f"{name}_bn")(x)
    if pool:
        x = layers.MaxPooling2D((2,2), name=None if not name else f"{name}_pool")(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout, name=None if not name else f"{name}_drop")(x)
    return x

def spatial_attention_module(feature_map, name='SAM'):
    # channel-wise pooling -> (H,W,1) each
    avg_pool = K.mean(feature_map, axis=-1, keepdims=True)
    max_pool = K.max(feature_map, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1, name=f'{name}_concat')([avg_pool, max_pool])  # (H,W,2)

    spatial_att = layers.Conv2D(1, (7,7), padding='same', activation='sigmoid', name=f'{name}_conv')(concat)  # (H,W,1)
    refined = layers.Multiply(name=f'{name}_mul')([feature_map, spatial_att])  # apply attention

    gap = layers.GlobalAveragePooling2D(name=f'{name}_gap')(refined)  # vector
    return refined, gap

def temporal_attention_module(lstm_seq, name='TAM'):
    # Project each timestep then compute scalar scores -> softmax -> weighted sum
    feature_dim = K.int_shape(lstm_seq)[-1]
    u = layers.TimeDistributed(layers.Dense(feature_dim, activation='tanh'), name=f'{name}_proj')(lstm_seq)
    scores = layers.TimeDistributed(layers.Dense(1), name=f'{name}_score')(u)  # (batch, timesteps, 1)
    scores = layers.Reshape((K.int_shape(scores)[1],), name=f'{name}_scores_reshape')(scores)  # (batch, timesteps)
    alphas = layers.Activation('softmax', name=f'{name}_alphas')(scores)  # (batch, timesteps)
    alphas_expanded = layers.Reshape((K.int_shape(alphas)[1], 1), name=f'{name}_alphas_expand')(alphas)
    weighted = layers.Multiply(name=f'{name}_weighted')([lstm_seq, alphas_expanded])
    context = layers.Lambda(lambda z: K.sum(z, axis=1), name=f'{name}_context')(weighted)  # (batch, features)
    return context, alphas

def build_model(input_shape=(SIZE, SIZE, 3),
                base_filters=BASE_FILTERS,
                blocks=BLOCKS,
                drop_rate=DROP_RATE,
                lstm_units=LSTM_UNITS,
                timesteps=TIMESTEPS,
                dense_units=DENSE_UNITS):
    inputs = layers.Input(shape=input_shape, name='input_image')

    x = inputs
    filters = base_filters
    for i in range(1, blocks+1):
        x = conv_block(x, filters=filters, pool=True, dropout=drop_rate, name=f'block{i}')
        filters *= 2

    feature_map = x  # (H, W, C)

    # SAM branch
    refined_map, sam_vec = spatial_attention_module(feature_map, name='SAM')

    # TAM branch: flatten -> reshape -> Bi-LSTM -> temporal attention
    flat = layers.Flatten(name='flatten')(feature_map)
    feat_dim = K.int_shape(flat)[-1]

    features_per_step = feat_dim // timesteps
    if features_per_step < 1:
        raise ValueError(f"timesteps ({timesteps}) too large for flattened feature dim ({feat_dim}).")

    use_len = features_per_step * timesteps
    if use_len != feat_dim:
        flat_trim = layers.Lambda(lambda z: z[:, :use_len], name='flatten_trim')(flat)
    else:
        flat_trim = flat

    seq = layers.Reshape((timesteps, features_per_step), name='reshape_for_lstm')(flat_trim)
    lstm_out = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True), name='bilstm')(seq)

    tam_vec, tam_alphas = temporal_attention_module(lstm_out, name='TAM')

    # combine
    combined = layers.Concatenate(name='concat')([sam_vec, tam_vec])
    x = layers.Dense(dense_units, activation='relu', name='fc1')(combined)
    x = layers.Dropout(0.5, name='drop_fc1')(x)
    out = layers.Dense(1, activation='sigmoid', name='out')(x)

    model = models.Model(inputs=inputs, outputs=out, name='CNN_SAM_LSTM_TAM')
    return model, tam_alphas

# ---------------------------
# Training & evaluation helpers
# ---------------------------
def train_and_evaluate():
    X_train, X_test, y_train, y_test = build_dataset(IMAGE_DIR)

    model, tam_alphas_layer = build_model(input_shape=X_train.shape[1:],
                                          base_filters=BASE_FILTERS,
                                          blocks=BLOCKS,
                                          drop_rate=DROP_RATE,
                                          lstm_units=LSTM_UNITS,
                                          timesteps=TIMESTEPS,
                                          dense_units=DENSE_UNITS)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # callbacks
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    checkpoint = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        callbacks=[reduce_lr, early_stop, checkpoint],
                        verbose=1)

    # save final model (checkpoint will have best)
    model.save(MODEL_SAVE_PATH)

    # Evaluate: ROC / Youden threshold
    y_probs = model.predict(X_test).ravel()
    fpr, tpr, thresh = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    youden = tpr - fpr
    ix = np.argmax(youden)
    best_thresh = thresh[ix]
    print(f"AUC = {roc_auc:.4f}, Best (Youden) thresh = {best_thresh:.4f}")

    y_pred = (y_probs >= best_thresh).astype(int)
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

    # plot ROC
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1],'--', linewidth=0.7)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend(); plt.grid(True)
    plt.show()

    # Plot training curves
    plt.figure(); plt.plot(history.history['loss'], label='train_loss'); plt.plot(history.history['val_loss'], label='val_loss'); plt.legend(); plt.title('Loss')
    plt.figure(); plt.plot(history.history['accuracy'], label='train_acc'); plt.plot(history.history['val_accuracy'], label='val_acc'); plt.legend(); plt.title('Accuracy')
    plt.show()

    return model, (X_test, y_test), best_thresh, tam_alphas_layer

# ---------------------------
# Extract temporal attention weights for samples
# ---------------------------
def get_temporal_attention_weights(trained_model, tam_alphas_layer, sample_images):
    """
    Build a small model that outputs the tam_alphas tensor for given input images.
    tam_alphas_layer: name or symbolic tensor returned by build_model; here it's the symbolic layer we returned.
    sample_images: numpy array of images shape (N, H, W, C)
    """
    # Identify the layer by name (the tam_alphas symbol is a Keras tensor in the returned tuple)
    # To make it robust, we will search for a layer named 'TAM_alphas' (constructed as f'{name}_alphas')
    # But we returned tam_alphas_layer object earlier: we can construct a model that outputs it directly
    # If tam_alphas_layer is a Keras tensor, do:
    try:
        att_model = models.Model(inputs=trained_model.input, outputs=[trained_model.get_layer('TAM_alphas').output])
    except Exception:
        # fallback: try to find any layer with 'alphas' in name
        layer = None
        for lyr in trained_model.layers:
            if 'alphas' in lyr.name:
                layer = lyr
                break
        if layer is None:
            raise RuntimeError("Could not find a temporal attention 'alphas' layer in the built model.")
        att_model = models.Model(inputs=trained_model.input, outputs=[layer.output])

    alphas = att_model.predict(sample_images)
    # alphas shape: (N, timesteps)
    return alphas

# ---------------------------
# Run
# ---------------------------
if __name__ == '__main__':
    model, test_data, best_thresh, tam_alphas_layer = train_and_evaluate()
    X_test, y_test = test_data

    # Example: extract attention for first 8 test images (if you want)
    nshow = min(8, len(X_test))
    try:
        alphas = get_temporal_attention_weights(model, tam_alphas_layer, X_test[:nshow])
        print("Temporal attention weights (first samples):\n", alphas)
    except Exception as e:
        print("Could not extract temporal attention weights:", e)

    print(f"Model saved to: {MODEL_SAVE_PATH}")
