from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

def get_input(path):

    image = np.load(("{}").format(path))["x"]
    return image

def combine_generator(gen1, gen2):
    while True:
        image = gen1.next()
        image2 = gen2.next()
        yield (image[:,:,:,0].reshape([image.shape[0], 512, 512, 1]), image2[:,:,:,0].reshape([image2.shape[0], 512, 512, 1]))

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

            #rInt = np.random.randint(1, input.shape[0], 1)
            #input = input[rInt, :, :]
            #output = output[rInt, :, :]

            input = inputO.reshape([1, 512, 512])
            output = outputO.reshape([1, 512, 512])

            inn = datagen.flow(input.reshape(1, 512, 512, 1), seed=2)
            out = datagen.flow(output.reshape(1, 512, 512, 1), seed=2)

            zipper = combine_generator(inn, out)

            for k in range(3):
                data = zipper.next()
                input = data[0]
                output = data[1]

                #mask = output > 0.5
                #output = mask.astype(int)

                # Debug
                if False:
                    for i in range(1):
                        fig = plt.figure(figsize=(8, 8))
                        fig.add_subplot(1, 2, 1)
                        plt.imshow(data[0][i, :, :, 0].reshape(512, 512))
                        fig.add_subplot(1, 2, 2)
                        plt.imshow(data[1][i, :, :, 0].reshape(512, 512))
                        plt.show()

                batch_input += [input]
                batch_output += [output]

                size += input.shape[0]

        batch_x = np.array(batch_input).reshape(size, 512, 512, 1)
        batch_y = np.array(batch_output).reshape(size, 512, 512, 1)

        yield (batch_x, batch_y)

validation = []
for filename in os.listdir("compressed_validation_individual"):
    validation.append(("compressed_validation_individual/{}").format(filename))

model = load_model('best_v3.sav')

if True:
    with open("history/trainHistoryDict", "rb") as input_file:
        history = pickle.load(input_file)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
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
            pred = model.predict(im.reshape(1,512,512,1))
            cond = pred > 0.5
            pred = cond.astype(int)
            if 1 in pred:
                fig=plt.figure(figsize=(8, 8))
                fig.add_subplot(1,3,1)
                plt.imshow(im.reshape(512, 512))
                fig.add_subplot(1,3,2)
                plt.imshow(og[1][i,:,:,0].reshape(512, 512))
                fig.add_subplot(1,3,3)
                plt.imshow(pred.reshape(512, 512))
                plt.show()
