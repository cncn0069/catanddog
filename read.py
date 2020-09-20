from keras.preprocessing.image import ImageDataGenerator
from keras import models
import os
import learn

original_dastaset_dir = './datasets/cats_and_dogs/train'
base_dir  = './datasets/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')


for data_batch, labels_batch in train_generator:
    print('베치 데이터 크기:', data_batch.shape)
    print('베치 데이터 크기:', labels_batch.shape)
    break

model = learn.model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data = validation_generator,
    validation_steps=50
)

model.save('cats_and_dogs_small.h5')



