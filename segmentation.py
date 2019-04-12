import keras
import os
import numpy as np
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.utils import class_weight
from keras import backend as K

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

def combine_generator(gen1, gen2):
    while True:
        image = gen1.next()
        image2 = gen2.next()
        cond = image2 > 68
        image2 = cond.astype(int)
        yield (image[:,:,:,0].reshape([image.shape[0], 512, 512, 1]), image2[:,:,:,0].reshape([image2.shape[0], 512, 512, 1]))

def get_input(path):

    image = np.load(("compressed_data/{}").format(path))["x"]
    return image

def get_output(path):

    image = np.load(("compressed_data/{}").format(path))["y"]
    return image

def custom_generator(files, batch_size = 1):

    while True:

        batch_path = np.random.choice(files, batch_size)

        batch_input = []
        batch_output = []

        size = 0

        for input_path in batch_path:

            input = get_input(input_path)
            output = get_output(input_path)

            #preprocessing

            batch_input += [input]
            batch_output += [output]

            size += input.shape[0]


        batch_x = np.array(batch_input).reshape(size, 512, 512, 1)
        batch_y = np.array(batch_output).reshape(size, 512, 512, 1)

        yield(batch_x, batch_y)



model = keras.models.Sequential()

baseFilter = 2
epochs = 16

inputs = Input((512, 512, 1))
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
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(baseFilter*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
conv5 = Conv2D(baseFilter*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

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
conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

model = Model(input=inputs, output=conv10)

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

args = dict(featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
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


w = [1, 25]
batch_size = 16

names = []

for filename in os.listdir("compressed_data"):
    names.append(filename)

if False:
    for i in range(50):
        img = custom_generator(names).next()
        print(img[0].shape)
        fig=plt.figure(figsize=(8, 8))
        for i in range(img[0].shape[0]):
            if i < 25:
                if i % 2 == 0:
                    fig.add_subplot(5,5,i+1)
                    plt.imshow(img[0][i, :, :, 0])
                    #print(img[0][i, :, :, 0])
                else:
                    fig.add_subplot(5,5,i+1)
                    plt.imshow(img[1][i-1, :, :, 0])
                    #print(np.unique(img[1][i-1, :, :, 0]))
        plt.show()

history = model.fit_generator(custom_generator(names), batch_size, epochs=epochs, class_weight=w)

if True:
    im = x_train[405, :, :, 0]
    print(im.shape)
    im = im.reshape(1, 512, 512, 1)
    print(im.shape)
    pred = model.predict(im)
    print(pred.shape)
    plt.imshow(y_train[405,:,:,0].reshape(512, 512))
    plt.show()
    plt.imshow(pred.reshape(512, 512))
    plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['vgg_train', 'vgg_val', 'my_train', 'my_val'], loc='upper left')
plt.show()
