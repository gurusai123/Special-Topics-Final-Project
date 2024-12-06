
import tensorflow as tf
import numpy as np 
import pandas as pd 
import tensorflow as tf
import os

import cv2
from PIL import Image

from sklearn import metrics 
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, BatchNormalization, Flatten
from keras.models import Model, Sequential
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras import backend as K
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image
import re

import time
import datetime



# Enable GPU globally
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.set_visible_devices(gpus, 'GPU')

        print(f"Using {len(gpus)} GPU(s): {gpus}")
    except RuntimeError as e:
        print(f"Error during GPU setup: {e}")
else:
    print("No GPU detected, running on CPU.")
    

def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today()



# To plot performance of the model
def plot_performance(history=None, figure_directory=None, ylim_pad=[0, 0]):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    y1 = history.history['accuracy']
    y2 = history.history['val_accuracy']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]


    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()


data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Reading and preprocessing images
for i in range(classes):
    path = os.path.join(r'C:\Users\gurus\Downloads\archive (1)','Train',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '/'+ a)
            image = image.resize((32,32))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Checking the data shapes
print(data.shape, labels.shape)

# Splitting the dataset into training and test
X_train, val_images, y_train, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Displaying the shape of the data after the split
print(X_train.shape, val_images.shape, y_train.shape, val_labels.shape)


train_images, train_labels = X_train, y_train


len(train_images),len(val_labels)

# Convert labels to one-hot encoded vectors
y_train = to_categorical(y_train, num_classes=43)
val_labels = to_categorical(val_labels, num_classes=43)


# Defining VGGNet Model
from keras.layers import Input
def VGGNet():
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), 2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), 2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    
    layer_flat = Flatten()(x)
    x = Dense(128, activation='relu')(layer_flat)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    layer_out = Dense(43, activation='softmax')(x)

    model = Model(input_img, layer_out)
    
    return model


# Defining LeNet Model
def LeNet():
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(1, (5, 5), activation='relu')(input_img)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(6, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    layer_flat = Flatten()(x)

    x = Dense(120, activation='relu')(layer_flat)

    x = Dense(84, activation='relu')(x)

    layer_out = Dense(43, activation='softmax')(x)

    model = Model(input_img, layer_out)
    
    return model


# Defining a function to Train and Save the model
def TrainModel(model, modelName, train_images, train_labels, val_images, val_labels,
               epochs=30, batch_size=64, disable_early_stopping=False):
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
    
    if disable_early_stopping:
        callbacks=[]
    else:
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=5)]
    
    model_train = model.fit(train_images, train_labels,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(val_images, val_labels),
                            shuffle=True,
                            callbacks=callbacks)
    

    model.save('models/' + modelName + '.h5')
    K.clear_session()
    loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']
    plt.figure()
    plt.plot(val_loss, 'r', label='Val loss')
    plt.plot(loss, 'bo', label='Training loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('training_and_valid_loss_' + modelName + '.png')
    plt.show()
    return model_train


# Defining function to Calculate Error
def CalculateError(real, predicted):
    err = 0
    for i in range(len(predicted)):
        if real[i] != predicted[i]:
            err = err + 1
    err_percent = err/len(predicted)*100
    accuracy = 100 - err_percent
    print('total test data: %d wrong predicted: %d' % (len(predicted), err))
    print('error: %f' % err_percent )
    print('accuracy: %f' % accuracy)
    return err


# Defining Function to predict and call the model
def Predict(modelPath, test_images, test_labels):
    model = load_model(modelPath)
    predictions = model.predict(test_images)
    pred = []
    real = []
    for i in range(len(predictions)):
        pred.append(np.where(predictions[i] == predictions[i].max())[0])
        real.append(np.where(test_labels[i] == test_labels[i].max())[0])

    CalculateError(real, pred)
    return pred


def getLabel(one_hot_encoded):
    return np.where(one_hot_encoded == one_hot_encoded.max())[0]


VGGNetModel = VGGNet()
VGGNetModel.summary()


abcd = TrainModel(VGGNetModel, "vggnet", train_images, train_labels, val_images, val_labels)


np.shape(val_images),np.shape(train_images)


print(abcd.history)


print(Predict("models/vggnet.h5", test_images, test_labels))
print(Predict("models/vggnet.h5", train_images, train_labels))


lenet = LeNet()
lenet.summary()


lenet_model = TrainModel(VGGNetModel, "lenet", train_images, train_labels, val_images, val_labels)


from sklearn.metrics import accuracy_score

# Importing the test dataset
y_test = pd.read_csv('../input/gtsrb-german-traffic-sign/Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

# Retreiving the images
with tf.device('/GPU:0'):
    for img in imgs:
        image = Image.open('../input/gtsrb-german-traffic-sign/'+img)
        image = image.resize([30, 30])
        data.append(np.array(image))

X_test=np.array(data)

with tf.device('/GPU:0'):
    pred = np.argmax(model.predict(X_test), axis=-1)


from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))


print(Predict("models/lenet5.h5", test_images, test_labels))
print(Predict("models/lenet5.h5", train_images, train_labels))



data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Loading the Retrieving the Adversarial images and their labels 
for i in range(classes):
    path = os.path.join(r'C:\Users\gurus\Downloads\adversarial','info.csv',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '/'+ a)
            image = image.resize((32,32))
            image = np.array(image)
            #sim = Image.fromarray(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

adv_ep1_images, adv_ep1_labels = data,labels

adv_training_images = []
adv_training_labels = []
adv_training_images.extend(adv_ep1_images[:500])
adv_training_labels.extend(adv_ep1_labels[:500])
adv_training_images.extend(adv_ep10_images[:500])
adv_training_labels.extend(adv_ep10_labels[:500])
adv_training_images.extend(adv_ep15_images[:500])
adv_training_labels.extend(adv_ep15_labels[:500])

adv_test_images = []
adv_test_labels = []
adv_test_images.extend(adv_ep1_images[500:])
adv_test_labels.extend(adv_ep1_labels[500:])
adv_test_images.extend(adv_ep10_images[500:])
adv_test_labels.extend(adv_ep10_labels[500:])
adv_test_images.extend(adv_ep15_images[500:])
adv_test_labels.extend(adv_ep15_labels[500:])

new_training_images = []
new_training_labels = []
new_test_images = []
new_test_labels = []

new_training_images.extend(train_images)
new_training_images.extend(adv_training_images)
new_training_labels.extend(train_labels)
new_training_labels.extend(adv_training_labels)

new_test_images.extend(test_images)
new_test_images.extend(adv_test_images)
new_test_labels.extend(test_labels)
new_test_labels.extend(adv_test_labels)


#Training lenet on blend of clean and adveserial images
LeNetAdvModel = LeNet()
TrainModel(LeNetAdvModel, "lenet5_adv", np.asarray(new_training_images),
           np.asarray(new_training_labels), val_images, val_labels, epochs=20, disable_early_stopping=True)

p_lenet_adv_test = Predict(r"C:\Users\gurus\Downloads\lenet5_adv.h5", np.asarray(new_test_images), np.asarray(new_test_labels))

p_lenet_adv_train = Predict(r"C:\Users\gurus\Downloads\lenet5_adv.h5", np.asarray(adv_test_images), np.asarray(adv_test_labels))

#Training VGGNet on blend of clean and adveserial images
VGGNetAdvModel = VGGNet()
TrainModel(VGGNetAdvModel, "vggnet_adv", np.asarray(new_training_images2),
           np.asarray(new_training_labels2), val_images, val_labels, epochs=30, disable_early_stopping=True)

p_vgg_adv_test = Predict(r"C:\Users\gurus\Downloads\vggnet_adv.h5", np.asarray(new_test_images2), np.asarray(new_test_labels2))

p_vgg_adv_train = Predict(r"C:\Users\gurus\Downloads\vggnet_adv.h5", np.asarray(new_training_images2), np.asarray(new_training_labels2))

get_ipython().system('jupyter nbconvert --to script traffic-signs-image-classification-gpu-enable.ipynb')