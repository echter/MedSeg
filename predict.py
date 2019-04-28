import matplotlib
matplotlib.use('Agg')

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
from keras import backend as K
from skimage import measure
import cv2
import scipy.misc

imageSize = 384

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

import keras.losses
keras.losses.generalized_dice_loss_w = generalized_dice_loss_w

def get_input(path):

    image = np.load(("{}").format(path))["x"]
    return image

def combine_generator(gen1, gen2):
    while True:
        image = gen1.next()
        image2 = gen2.next()
        yield (image[:,:,:,0].reshape([image.shape[0], 256, 256, 1]), image2[:,:,:,0].reshape([image2.shape[0], 256, 256, 1]))

def get_output(path):

    image = np.load(("{}").format(path))["y"]
    return image

def custom_generator(files, batch_size=1):
    # TODO only keep data which has more than x pixels. Currently some images with like 0 pixels are allowed which is bad

    args = dict(rotation_range=0,
        width_shift_range=0.01,
        height_shift_range=0.01,
        # rescale= 1. / 255,
        shear_range=0.05,
        zoom_range=0.1,
        vertical_flip=True,
        fill_mode='nearest')

    datagen = ImageDataGenerator(**args)

    while True:

        batch_path = np.random.choice(files, batch_size)

        batch_input = []
        batch_output = []

        size = 0

        # TODO make this select a bunch of random slices isntead of the whole set from 1 image
        for input_path in batch_path:

            inputO = get_input(input_path)
            outputO = get_output(input_path)

            input = inputO.reshape([1, imageSize, imageSize])
            output = outputO.reshape([1, imageSize, imageSize])

            batch_input += [input]
            batch_output += [output]

            size += input.shape[0]

        batch_x = np.array(batch_input).reshape(size, imageSize, imageSize, 1)
        batch_y = np.array(batch_output).reshape(size, imageSize, imageSize, 1)

        yield (batch_x, batch_y)

validation = []
for filename in os.listdir("compressed_validation_individual"):
    validation.append(("compressed_validation_individual/{}").format(filename))

model = load_model('best_v5.sav')

if True:
    with open("history/trainHistoryDict", "rb") as input_file:
        history = pickle.load(input_file)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['vgg_train', 'vgg_val', 'my_train', 'my_val'], loc='upper left')
    plt.savefig("v7.png")
    #plt.show()

if False:
    for k in range(50):
        og = custom_generator(validation).next()
        for i in range(og[0].shape[0]):
            im = og[0][i,:,:,0]
            pred = model.predict(im.reshape(1,imageSize,imageSize,1))

            cond = pred > np.amax(pred) * 0.8
            pred = cond.astype(int)

            pred = np.array(pred.reshape(imageSize, imageSize))
            pred = pred.astype(np.uint8)

            contours, hierarchy = cv2.findContours(pred, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            area=0
            contourList = 0
            for c in contours:
                if (cv2.contourArea(c) > area):
                    area = cv2.contourArea(c)
                    contourList = c

            fig=plt.figure(figsize=(16, 16))

            ax1 = fig.add_subplot(1,4,1)
            plt.imshow(im.reshape(imageSize, imageSize))
            ax1.set_title("Input")

            ax2 = fig.add_subplot(1,4,2)
            plt.imshow(og[1][i,:,:,0].reshape(imageSize, imageSize))
            ax2.set_title("Label")

            ax4 = fig.add_subplot(1,4,4)
            plt.imshow(pred.reshape(imageSize, imageSize))
            ax4.set_title("Prediction")

            ax3 = fig.add_subplot(1,4,3)
            im = np.zeros((imageSize, imageSize))
            if (np.amax(pred) > 0):
                cv2.fillPoly(im, pts=[contourList], color=(255,255,255))
            plt.imshow(im)
            ax3.set_title("Post processed")

            plt.show()
