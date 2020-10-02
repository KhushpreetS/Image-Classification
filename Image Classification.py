#Image classification for two classes
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.layers import Conv2D , MaxPooling2D , Dropout
from keras.layers import Dense , Flatten 
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder

from imutils import paths
import numpy as np
import argparse
import cv2
import os

#First class data
imagePaths = list(paths.list_images(r"C:\Users\Desktop\Image classification\class1"))

data = []
labels = []

for imagePath in imagePaths:
    
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    data.append(image)
    labels.append(label)

#Second class data  
imagePaths2 = list(paths.list_images(r"C:\Users\Desktop\Image classification\class2"))
for imagePath in imagePaths2:
   
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    data.append(image)
    labels.append(label)
 
#normalizing the data in list 
data = np.array(data) / 255.0
labels = np.array(labels)

#associating labels
lb = LabelEncoder()
labels = lb.fit_transform(labels)
print(labels)

#data split
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)

#model creation
model = Sequential()

model.add(Conv2D(32, (3,3), activation = 'relu', input_shape=(128,128,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128 , (3,3) , activation = 'relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile( loss = "binary_crossentropy" , optimizer  = 'adam' , metrics = ['categorical_accuracy'])

model.fit(X_train,Y_train,epochs = 10)
loss , accuracy = model.evaluate(X_test , Y_test )

model_json = model.to_json()
with open(r"C:\Users\model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
filepath =r"C:\Users\Desktop\Image classification\model.h5"

#saving the model
model.save_weights(filepath)
print("Saved model to disk")

#testing the model
test=[]
image = cv2.imread(r"C:\Users\Desktop\Image classification\test\IM-1294-0001-0001.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (128, 128))
test.append(image)
test = np.array(test) / 255.0
model.predict(test)
