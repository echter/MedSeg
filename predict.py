from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

def get_input(path):

    image = np.load(("{}").format(path))["x"]
    return image

def get_output(path):

    image = np.load(("{}").format(path))["y"]
    return image

def custom_generator(files, batch_size = 1):

    while True:

        batch_path = np.random.choice(files, batch_size)

        batch_input = []
        batch_output = []

        size = 0

        # TODO make this select a bunch of random slices isntead of the whole set from 1 image
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

validation = []
for filename in os.listdir("compressed_validation"):
    validation.append(("compressed_validation/{}").format(filename))

model = load_model('best.sav')

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

if False:
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
