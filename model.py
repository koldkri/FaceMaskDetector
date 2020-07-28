import tensorflow as tf
import os
import random
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

try:
    os.mkdir('C:/Users/dell/IdeaProjects/project_opencv/data/training/')
    os.mkdir('C:/Users/dell/IdeaProjects/project_opencv/data/testing/')
    os.mkdir('C:/Users/dell/IdeaProjects/project_opencv/data/training/with_mask/')
    os.mkdir('C:/Users/dell/IdeaProjects/project_opencv/data/training/without_mask/')
    os.mkdir('/C:/Users/dell/IdeaProjects/project_opencv/data/testing/with_mask/')
    os.mkdir('C:/Users/dell/IdeaProjects/project_opencv/data/testing/without_mask/')

except OSError:
    pass



def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):

    source_list = os.listdir(SOURCE)
    source_list=random.sample(source_list,len(source_list))
    train=source_list[:int(len(source_list)*SPLIT_SIZE)]
    test=source_list[int(len(source_list)*SPLIT_SIZE):]

    for tr in train:
        if os.path.getsize("{}{}".format(SOURCE,tr))!=0:
            source=os.path.join(SOURCE,tr)
            copyfile(source,'{}{}'.format(TRAINING,tr))

    for te in test:
        if os.path.getsize("{}{}".format(SOURCE,te))!=0:
            source=os.path.join(SOURCE,te)
            copyfile(source,'{}{}'.format(TESTING,te))

MASK_SOURCE_DIR = "C:/Users/dell/IdeaProjects/project_opencv/data/with_mask/"
TRAINING_MASK_DIR = "C:/Users/dell/IdeaProjects/project_opencv/data/training/with_mask/"
TESTING_MASK_DIR = "C:/Users/dell/IdeaProjects/project_opencv/data/testing/with_mask/"
NOMASK_SOURCE_DIR = "C:/Users/dell/IdeaProjects/project_opencv/data/without_mask/"
TRAINING_NoMASK_DIR = "C:/Users/dell/IdeaProjects/project_opencv/data/training/without_mask/"
TESTING_NoMASK_DIR = "C:/Users/dell/IdeaProjects/project_opencv/data/testing/without_mask/"

split_size = .92
split_data(MASK_SOURCE_DIR, TRAINING_MASK_DIR, TESTING_MASK_DIR, split_size)
split_data(NOMASK_SOURCE_DIR, TRAINING_NoMASK_DIR, TESTING_NoMASK_DIR, split_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid'),
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = 'C:/Users/dell/IdeaProjects/project_opencv/data/training/'
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,target_size=(150,150),batch_size=10,class_mode='binary')

VALIDATION_DIR = 'C:/Users/dell/IdeaProjects/project_opencv/data/testing/'
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,target_size=(150,150),batch_size=10,class_mode='binary')

history = model.fit_generator(train_generator,
                              epochs=2,
                              verbose=1,
                              validation_data=validation_generator)

