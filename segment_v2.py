import keras
import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from sklearn.utils import class_weight
from keras import backend as K
import tensorflow as tf
import random
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

imageSize = 384

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    #dz = np.zeros_like(dx)

    #x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def combine_generator(gen1, gen2):
    while True:
        image = gen1.next()
        image2 = gen2.next()
        yield (image[:,:,:,0].reshape([image.shape[0], imageSize, imageSize, 1]), image2[:,:,:,0].reshape([image2.shape[0], imageSize, imageSize, 1]))

def get_input(path):

    image = np.load(("{}").format(path))["x"]
    return image

def get_output(path):

    image = np.load(("{}").format(path))["y"]
    return image

def custom_generator(files, batch_size=16):
    # TODO only keep data which has more than x pixels. Currently some images with like 0 pixels are allowed which is bad

    args = dict(rotation_range=0,
        width_shift_range=0.01,
        height_shift_range=0.01,
        # rescale= 1. / 255,
        shear_range=0.05,
        zoom_range=0.05,
        vertical_flip=True,
        fill_mode='nearest')

    datagen = ImageDataGenerator(**args)

    while True:

        batch_path = []#np.random.choice(files, batch_size)

        for i in range(batch_size):
            batch_path.append(random.choice(files))

        batch_input = []
        batch_output = []

        size = 0

        # TODO make this select a bunch of random slices isntead of the whole set from 1 image
        for input_path in batch_path:

            randomState = np.random.randint(0, 1000, 1)
            #print(input_path)
            #print(input_path)
            inputO = get_input(input_path)
            outputO = get_output(input_path)

            input = inputO.reshape(imageSize, imageSize)
            input = input.astype(np.float32)
            output = outputO.reshape(imageSize, imageSize)
            output = output.astype(np.float32)

            input_transform = elastic_transform(input, input.shape[1] * 2, input.shape[1] * 0.08, input.shape[1] * 0.08, random_state=np.random.RandomState(randomState))
            output_transform = elastic_transform(output, output.shape[1] * 2, output.shape[1] * 0.08, output.shape[1] * 0.08, random_state=np.random.RandomState(randomState))

            mask = output_transform > 0.5
            output_transform = mask.astype(int)

            batch_input += [input_transform]
            batch_output += [output_transform]
            size += 1

        batch_x = np.array(batch_input).reshape(size, imageSize, imageSize, 1)
        batch_y = np.array(batch_output).reshape(size, imageSize, imageSize, 1)

        yield (batch_x, batch_y)

def validation_custom_generator(files, batch_size=16):

    while True:

        batch_path = []#np.random.choice(files, batch_size)

        for i in range(batch_size):
            batch_path.append(random.choice(files))

        batch_input = []
        batch_output = []

        size = 0

        # TODO make this select a bunch of random slices isntead of the whole set from 1 image
        for input_path in batch_path:

            #print(input_path)

            input = get_input(input_path)
            output = get_output(input_path)

            batch_input += [input]
            batch_output += [output]
            size += 1

        batch_x = np.array(batch_input).reshape(size, imageSize, imageSize, 1)
        batch_y = np.array(batch_output).reshape(size, imageSize, imageSize, 1)

        yield (batch_x, batch_y)

def generalized_dice_loss_w(y_true, y_pred):

    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = K.sum(numerator,(0,1,2))
    numerator = K.sum(numerator)

    denominator = y_true+y_pred
    denominator = K.sum(denominator,(0,1,2))
    denominator = K.sum(denominator)

    gen_dice_coef = numerator/denominator

    return 1-2*gen_dice_coef

x_train = []
y_train = []

counter = 0
limit = 2000

weightTest1 = 0
weightTest0 = 0

for filename in os.listdir("compressed_data_individual"):
    if counter < limit:
        print(("Reading image {} of {}.").format(counter+1, limit))
        y_train.append(np.load(("compressed_data_individual/{}").format(filename))["y"])
    counter+=1

x_train = np.array(x_train)
y_train = np.array(y_train).flatten()

print(np.unique(y_train))

model = keras.models.Sequential()

baseFilter = 32
dropout = 0.65

# Keras U-Net implementation
# Base model taken from https://github.com/zhixuhao/unet/blob/master/model.py
inputs = Input((imageSize, imageSize, 1))
conv1 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
conv1 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
conv2 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
conv3 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
conv4 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
drop4 = Dropout(dropout)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(baseFilter*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
conv5 = Conv2D(baseFilter*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
drop5 = Dropout(dropout)(conv5)

up6 = Conv2D(baseFilter*8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
merge6 = concatenate([drop4, up6], axis=3)
conv6 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
conv6 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

up7 = Conv2D(baseFilter*4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
conv7 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

up8 = Conv2D(baseFilter*2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
conv8 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

up9 = Conv2D(baseFilter, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
conv9 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

model = Model(input=inputs, output=conv10)

model.compile(optimizer=Adam(lr=1e-5, decay=1e-9), loss=generalized_dice_loss_w, metrics=['accuracy'])#, decay=(1e-6))

args = dict(featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)

if False:
    seed = 1

    #x_generator.fit(x_train, augment=True, seed=seed)
    #y_generator.fit(y_train, augment=True, seed=seed)
    x_generator = ImageDataGenerator(**args)
    y_generator = ImageDataGenerator(**args)
    x_generator = x_generator.flow_from_directory('data/img', class_mode=None, seed=seed, target_size=(512,512))
    y_generator = y_generator.flow_from_directory('data/mask', class_mode=None, seed=seed, target_size=(512,512))
    train_generator = combine_generator(x_generator, y_generator)

    x_val = ImageDataGenerator(**args)
    y_val = ImageDataGenerator(**args)
    x_val = x_val.flow_from_directory('data/val_img', class_mode=None, seed=seed, target_size=(512,512))
    y_val = y_val.flow_from_directory('data/val_mask', class_mode=None, seed=seed, target_size=(512,512))
    val_generator = combine_generator(x_val, y_val)

#class_weights = [1, 100]

batch_size = 4
epochs = 512

names = []
for filename in os.listdir("compressed_data_individual"):
    names.append(("compressed_data_individual/{}").format(filename))

steps = len(names) / batch_size

if False:
    for i in range(50):
        img = validation_custom_generator(names, batch_size=1).next()
        #img = custom_generator(names, batch_size=1).next()
        print(img[0].shape)
        fig=plt.figure(figsize=(8, 8))
        for i in range(img[0].shape[0]):
            if i < 25:
                #if i % 2 == 0:
                fig.add_subplot(1,2,1)
                plt.imshow(img[0][i, :, :, 0])
                    #print(img[0][i, :, :, 0])
                #else:
                fig.add_subplot(1,2,2)
                plt.imshow(img[1][i-1, :, :, 0])
                    #print(np.unique(img[1][i-1, :, :, 0]))
        plt.show()
if False:
    while True:
        img = custom_generator(names).next()

        input = img[0][0,:,:].reshape(imageSize, imageSize)
        im_merge_t = elastic_transform(input, input.shape[1] * 2, input.shape[1] * 0.08, input.shape[1] * 0.08, random_state=np.random.RandomState(2))

        fig=plt.figure(figsize=(8, 8))
        fig.add_subplot(1,2,1)
        plt.imshow(im_merge_t)

        input = img[1][0,:,:].reshape(imageSize, imageSize)
        input = input.astype(np.float32)
        im_merge_t = elastic_transform(input, input.shape[1] * 2, input.shape[1] * 0.08, input.shape[1] * 0.08, random_state=np.random.RandomState(2))

        fig.add_subplot(1,2,2)
        plt.imshow(im_merge_t)
        plt.show()

validation = []
for filename in os.listdir("compressed_validation_individual"):
    validation.append(("compressed_validation_individual/{}").format(filename))

#class_weights = [1, weightTest0/weightTest1]
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
print("="*80)
print(class_weight)
print("="*80)

checkpointer = ModelCheckpoint("best_v7.sav", monitor='val_loss',
    verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
earlyStopper = EarlyStopping(monitor='val_loss', min_delta=0,
    patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
history = model.fit_generator(custom_generator(names, batch_size=batch_size), steps,
    epochs=epochs, validation_data=validation_custom_generator(validation, batch_size=batch_size*16),
    validation_steps=2, shuffle=True, callbacks=[checkpointer, earlyStopper])

with open('history/trainHistoryDict_v7', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

if False:
    im = x_train[405, :, :, 0]
    print(im.shape)
    im = im.reshape(1, 512, 512, 1)
    print(im.shape)
    pred = model.predict(im)
    cond = pred > 0.5
    pred = cond.astype(int)
    print(pred.shape)
    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(1,3,1)
    plt.imshow(x_train[405,:,:,0].reshape(512, 512))
    fig.add_subplot(1,3,2)
    plt.imshow(y_train[405,:,:,0].reshape(512, 512))
    fig.add_subplot(1,3,3)
    plt.imshow(pred.reshape(512, 512))
    plt.show()
    print(np.unique(pred))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['vgg_train', 'vgg_val', 'my_train', 'my_val'], loc='upper left')
plt.show()

if True:
    for k in range(10):
        og = custom_generator(validation).next()
        for i in range(og[0].shape[0]):
            im = og[0][i,:,:,0]
            pred = model.predict(im.reshape(1,imageSize,imageSize,1))
            #print(np.unique(pred, return_counts=True))
            cond = pred > 0.5
            pred = cond.astype(int)
            #print(np.unique(pred, return_counts=True))
            #if 1 in pred:
            fig=plt.figure(figsize=(8, 8))
            fig.add_subplot(1,3,1)
            plt.imshow(im.reshape(imageSize, imageSize))
            fig.add_subplot(1,3,2)
            plt.imshow(og[1][i,:,:,0].reshape(imageSize, imageSize))
            fig.add_subplot(1,3,3)
            plt.imshow(pred.reshape(imageSize, imageSize))
            plt.show()
