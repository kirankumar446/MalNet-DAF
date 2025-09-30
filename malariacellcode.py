import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#plt.style.use('classic')
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Reshape, Conv2D, LSTM, Bidirectional, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ReduceLROnPlateau
import os
import cv2
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential

image_directory = "E:\data-kiran\Recovered Data MSI\B Tech Projects AITAM\Batch B9 2023-2024\images\cell_images"
SIZE = 150
dataset = []

label = []  

parasitized_images = os.listdir( image_directory + '/Parasitized/')
for i, image_name in enumerate(parasitized_images):    
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory +'/Parasitized/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0) 

uninfected_images = os.listdir(image_directory + '/Uninfected/')
for i, image_name in enumerate(uninfected_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + '/Uninfected/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)
from keras.utils import normalize
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

INPUT_SHAPE = (SIZE, SIZE, 3)

model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(SIZE, SIZE, 3)))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.25))
#model.add(Dense(1, activation="sigmoid"))

model.add(Reshape((1, -1)))
model.add(Bidirectional(LSTM(32, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(256, activation='tanh')))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

from keras.utils import plot_model 
plot_model(model, to_file='model.png')

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

history = model.fit(X_train, 
                         y_train, 
                         batch_size = 128, 
                         verbose = 1, 
                         epochs = 1,      
                         validation_data=(X_test,y_test),
                         shuffle = False,
                         callbacks=[reduce_lr]
                    )
model.save('malaria_model_10epochs.h5')  

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.plot(epochs, accuracy, 'y', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

n=18  
img = X_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) 
print("The prediction for this image is: ", model.predict(input_img))
print("The actual label for this image is: ", y_test[n])
#for accuracy
from keras.models import load_model


_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")
mythreshold=0.98
from sklearn.metrics import confusion_matrix
y_pred = (model.predict(X_test)>= mythreshold).astype(int)
cm=confusion_matrix(y_test, y_pred)  
print("Confusion Matrix:\n", cm)

#ROC
from sklearn.metrics import roc_curve
y_preds = model.predict(X_test).ravel()

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

import pandas as pd
i = np.arange(len(tpr)) 
roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thresholds' : pd.Series(thresholds, index=i)})
ideal_roc_thresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
print("Ideal threshold is: ", ideal_roc_thresh['thresholds']) 

from sklearn.metrics import auc
auc_value = auc(fpr, tpr)
print("Area under curve, AUC = ", auc_value)


# Making the Confusion Matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(12,12))
sns.set(font_scale=2.0)
sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, ax=ax)
ax.xaxis.set_ticklabels(['parasitized', 'uninfected']); ax.yaxis.set_ticklabels(['parasitized', 'uninfected'])
plt.title("Confusion Matrix")
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')