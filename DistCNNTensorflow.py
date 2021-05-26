#Import required libraries
import numpy as np
import os
import matplotlib.pyplot as plt #Matplotlib is used for generating graphs
import tensorflow as tf #tensorflow library for creating NNs 
from tensorflow.keras import datasets, layers, models

def training_data(train_dir, batch_size, img_height, img_width, colour_option):#Function for importing training images
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,#training data location
  color_mode=colour_option,#colour channels
  seed=123,#seed used to randomise dataset
  image_size=(img_height, img_width),#image shape
  batch_size=batch_size)#batch size to be used in NN
  return train_ds #returns Training data

def validation_data(val_dir, batch_size, img_height, img_width, colour_option):
  val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,#test data location
  color_mode=colour_option,#colour channels
  seed=123,#seed used to randomise dataset
  image_size=(img_height, img_width),#image shape
  batch_size=batch_size)#batch size to be used in NN
  return val_ds #returns Test data

def model_build(img_height, img_width, colour_chanels, hl1, hl2, hl3, hl4, hl5):#function for defining layers of CNN
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)#normalise input data between 0 and 1
    model = models.Sequential()#tensorflow model type
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, colour_chanels)))#convolution-1 with input shape. 16 layers and 3x3 kernel
    model.add(layers.MaxPooling2D((2, 2)))#maxpooling-1 2x2 kernel
    model.add(layers.Conv2D(30, (3, 3), activation='relu')) #reLU activation function used
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())#Flatten layers to 1x1 dimension. Input Layer
    model.add(layers.Dense(hl1, activation='relu'))#Hidden layer 1
    model.add(layers.Dense(hl2, activation='relu'))
    model.add(layers.Dense(hl3, activation='relu'))
    model.add(layers.Dense(hl4, activation='relu'))
    model.add(layers.Dense(hl5, activation='relu'))#Hidden layer 5
    model.add(layers.Dense(num_classes, activation='softmax'))#output layer, activation function softmax
    model.summary()#prints summary of CNN
    model.compile(#compiles NN with SGD optimiser set to 0.0001 learning rate
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])
    return model #returns compiled CNN model

#Line 49 must be commented out if using on PC. 
#On Cluster uncomment line 49 and set to "MultiWorkerMirroredStrategy()", On DGX set to "MirroredStrategy()"
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #uncomment to disable GPU support

time_start = tf.timestamp() #create timestamp to record start of program run
#HyperParameters
batch_size = 50 #number of images to be processed at one time
num_classes = 10 #number of classes in dataset
num_epochs = 50 #number of epochs CNN will perform
hl1 = 800 #Neurons per hidden layer
hl2 = 800
hl3 = 800
hl4 = 800
hl5 = 800
img_height = 28 #image dimensions
img_width = 28
colour_chanels = 1 #colour channels
#rgb for colour images grayscale for b&w
colour_option = 'grayscale' #colour mode
title = "Fashion "#dataset title
train_dir = "Datasets/Fashion/train" #training data location
val_dir = "Datasets/Fashion/test" #test data location
save_dir = "ModelSave/Fashion" #save directory for trained model

multi_training = training_data(train_dir, batch_size, img_height, img_width, colour_option) #perform function to fetch training data
multi_validation = validation_data(val_dir, batch_size, img_height, img_width, colour_option) #perform function to fetch test data

AUTOTUNE = tf.data.AUTOTUNE
multi_training = multi_training.cache().prefetch(buffer_size=AUTOTUNE)
multi_validation = multi_validation.cache().prefetch(buffer_size=AUTOTUNE)

#Line 81 must be commented out if using on PC and lines 86 to 92 must be against margin
#On Cluster or DGX uncomment line 81 and lines 83 to 88 must be indented 84 to 87 are nested
with strategy.scope():
  multi_model = model_build(img_height, img_width, colour_chanels, hl1, hl2, hl3, hl4, hl5)
  history = multi_model.fit(
    multi_training,#training data to use in CNN
    validation_data=multi_validation,#test data to use in CNN
    epochs=num_epochs,#epochs to use in CNN
    verbose=1#Level of detail in output
  )

time_end = tf.timestamp()#records end timestamp of CNN
evaluation = str(multi_model.evaluate(multi_validation,verbose = 2))#prints evaluation of CNN training
last_chars = evaluation[ 20: 27: 1]#extracts percentage from evaluation
print(last_chars)#prints percentage
total_time = "Total Time%8.2f Seconds" % float(time_end - time_start)#end timestamp minus start = total seconds for CNN to train 
print(total_time)#print time
#convert hidden layer neurons to strings
hl1_str = str(hl1)
hl2_str = str(hl2)
hl3_str = str(hl3)
hl4_str = str(hl4)
hl5_str = str(hl5)
#build file name for graph save file
file_extention = ".png"
file_name = title + total_time + last_chars

plt.plot(history.history['accuracy'], label='accuracy')#training results added to graph
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')#test results added to graph
plt.xlabel('Epoch')#x axis = epochs
plt.ylabel('Accuracy')#y axis = accuracy
plt.ylim([0.2, 1])#y axis display limits
plt.title(title + "Hidden layers " + "HL1 " + hl1_str + " HL2 " + hl2_str + " HL3 " + hl3_str + " HL4 " + hl4_str + "\nHL5 " + hl5_str + "\n" + total_time + " " + last_chars)#graph title
plt.legend(loc='lower right')#legend location
plt.tight_layout()
plt.savefig(file_name + file_extention)#save graph to file

multi_model.save(save_dir)#save trained CNN model