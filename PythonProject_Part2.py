#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from zipfile import ZipFile
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger


# In[2]:


os.getcwd()
#!unzip /Users/apple/Downloads/archive.zip
os.chdir('/Users/apple/Downloads/archive/Meta')
os.getcwd()


# In[3]:


#Loading the data
Images_data = [] #Loading Images to Images_data List
Images_labels = [] #Loading labels to Images_labels List
classes = 43 #Classes

for i in range(classes): #Looping all the classes 
    path = os.path.join('/Users/apple/Downloads/archive/Train',str(i))
    images = os.listdir(path)
    for a in images:#Looping through all the images
        image = Image.open(path + '/' + a)  #sending file path to the image variable
        image = image.resize((32,32)) #Resizing the images
        image = np.array(image)   #converted image to array
        Images_data.append(image) #Appending all the images to Images_data list
        Images_labels.append(i) #Appending all the labels to Image_labels list

Images_data = np.array(Images_data) #list to arrays
Images_labels = np.array(Images_labels) #list to arrays


# In[4]:


#printing the size of data and labels
print('Size of Images : ' ,Images_data.shape)
print('SIze of Labels : ' ,Images_labels.shape)


# In[5]:


#display the first image in the training data
plt.imshow(Images_data[105,:,:],cmap='gray')
plt.show()


# In[6]:


#Splitting the data into train and test
train_images,test_images,train_labels,test_labels = train_test_split(Images_data,Images_labels,test_size=0.2,random_state = 42)


# In[7]:


#printing the size of train and test data
print('train_images size : ' ,train_images.shape)
print('train_labels size : ' ,train_labels.shape)
print('test_images size :  ' ,test_images.shape)
print('test_labels size :  ' ,test_labels.shape)


# In[8]:


#change the labels from integer to one-hot encoding
train_labels = to_categorical(train_labels,43)
test_labels = to_categorical(test_labels,43)


# In[9]:


#Adding more dense layers
model = Sequential()
#hidden layer using activation relu
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=train_images.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu')) #adding more layers
model.add(MaxPooling2D(pool_size=(2, 2))) #adding more layers
model.add(Dropout(rate=0.25))
#Adding more Conv2D, Maxpooling, Dense layers
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) #adding more layers
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) #adding more layers
model.add(MaxPooling2D(pool_size=(2, 2))) #adding more layers
model.add(Dropout(rate=0.25))
#Flattening the model
model.add(Flatten())
model.add(Dense(256, activation='relu')) #adding more layers
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax')) #out layer


# In[10]:


#Compilation 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#Fitting or passing the data to the model
history = model.fit(train_images, train_labels, batch_size=256, epochs=5, verbose=1,validation_data=(test_images, test_labels))


# In[11]:


#Listing all the data in history (Call Backs)
print(history.history.keys())


# In[12]:


# Plotting the Accuracy for both training data and validation data using the history object.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['val_accuracy', 'accuracy'], loc='lower right')
plt.title('Accuracy data comparison')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.show()


# In[13]:


# Plotting the loss for both training data and validation data using the history object.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss data comparison')
plt.legend(['val_loss', 'loss'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()


# In[14]:


model.save('/Users/apple/Desktop/traffic_sign_board_detector.h5')
model.save('traffic_sign_board_detector.h5')
print(model.summary()) #summary of the model


# In[15]:


#Importing the Libraries
#Used PIL,glob,imageio instead of OpenCV
from PIL import Image 
import glob
import imageio
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps


# In[16]:


#getClassName function will map all the predicted labels to the Class Names.
def getClassName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'No passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'


# In[17]:


#reloading the model
from keras.models import load_model
model = load_model('traffic_sign_board_detector.h5')


# In[18]:


# import matplotlib.pyplot as plt
for image_path in glob.glob(r'/Users/apple/Downloads/image_album/*.png'):
    imgage1 = Image.open(image_path) #opening the every frame in the given path
    img = imgage1.resize((32, 32)) #resizing the images inorder to fit for model
    img = ImageOps.equalize(img,mask=None) #done equalize before passing to model
    im = np.array(img) #converted image to arrays 
    #im = im/255 #Normalising the values between 0 and 1 instead of 0 and 255
    im = im.reshape(1,32, 32, 3) #Reshaping the image of 1024 pixel to 32*32 1 dim
    predictions = model.predict([im])[0] #Passing the image to the model
    pred_classes = np.argmax(predictions,axis=0)
    ClassName = getClassName(pred_classes) 
    print("predicted class name is ",ClassName)
    plt.imshow(imgage1)
    plt.show()


# In[ ]:




