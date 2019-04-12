import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import json
import cv2
from matplotlib import pyplot as plt

def generate_train_data():
    img_again = nib.load('./imagesTr/lung_001.nii.gz')
    print(img_again.shape)

    with open('./dataset.json') as json_file:
        data = json.load(json_file)

    trainingData = data["training"]

    counter = 0
    for row in trainingData:
        if counter < 20:
            x_train = []
            y_train = []

            img = nib.load(trainingData[0]["image"]).get_fdata()
            label = nib.load(trainingData[0]["label"]).get_fdata()

            for i in range(img.shape[2]):
                x_train.append(img[:,:,i])
                y_train.append(label[:,:,i])

            counter += 1
            print("Reading slice: ", counter, " of: ", len(trainingData))

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            for i in range(x_train.shape[0]):
                #plt.imshow(x_train[i, :, :].reshape([512, 512]))
                #cv2.imshow("i", x_train[i, :, :].reshape([512, 512]))
                #plt.show()
                if 1 in y_train[i, :, :]:
                    plt.imsave(("data/img/0/{}_slice{}").format(row["image"].split("/")[-1].split(".")[0], i), x_train[i, :, :].reshape([512, 512]))
                    plt.imsave(("data/mask/0/{}_slice{}").format(row["image"].split("/")[-1].split(".")[0], i), y_train[i, :, :].reshape([512, 512]))
                    #print(x_train[i, :, :].reshape([512, 512]).shape)
                if i % 10 == 0:
                    print(i)

            #np.savez_compressed(("data/x_train/{}").format(row["image"].split("/")[-1]), data=x_train)
            #np.savez_compressed(("data/y_train/{}").format(row["image"].split("/")[-1]), data=y_train)

def generate_validation_data():
    img_again = nib.load('./imagesTr/lung_001.nii.gz')
    print(img_again.shape)

    with open('./dataset.json') as json_file:
        data = json.load(json_file)

    trainingData = data["training"]

    counter = 40
    for row in trainingData:
        if counter < 50:
            x_train = []
            y_train = []

            img = nib.load(trainingData[0]["image"]).get_fdata()
            label = nib.load(trainingData[0]["label"]).get_fdata()

            for i in range(img.shape[2]):
                x_train.append(img[:,:,i])
                y_train.append(label[:,:,i])

            counter += 1
            print("Reading slice: ", counter, " of: ", len(trainingData))

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            for i in range(x_train.shape[0]):
                #plt.imshow(x_train[i, :, :].reshape([512, 512]))
                #cv2.imshow("i", x_train[i, :, :].reshape([512, 512]))
                #plt.show()
                if 1 in y_train[i, :, :]:
                    plt.imsave(("data/val_img/0/{}_slice{}").format(row["image"].split("/")[-1].split(".")[0], i), x_train[i, :, :].reshape([512, 512]))
                    plt.imsave(("data/val_mask/0/{}_slice{}").format(row["image"].split("/")[-1].split(".")[0], i), y_train[i, :, :].reshape([512, 512]))
                    #print(x_train[i, :, :].reshape([512, 512]).shape)
                if i % 10 == 0:
                    print(i)

            #np.savez_compressed(("data/x_train/{}").format(row["image"].split("/")[-1]), data=x_train)
            #np.savez_compressed(("data/y_train/{}").format(row["image"].split("/")[-1]), data=y_train)

def convert_training_to_compressed_numpy():
    img_again = nib.load('./imagesTr/lung_001.nii.gz')
    print(img_again.shape)

    with open('./dataset.json') as json_file:
        data = json.load(json_file)

    trainingData = data["training"]

    counter = 0
    for row in trainingData:
        print(row)
        if counter < 41:
            x_train = []
            y_train = []

            img = nib.load(trainingData[0]["image"]).get_fdata()
            label = nib.load(trainingData[0]["label"]).get_fdata()

            for i in range(img.shape[2]):
                if 1 in label[:,:,i]:
                    x_train.append(img[:,:,i])
                    y_train.append(label[:,:,i])

            counter += 1
            print("Reading slice: ", counter, " of: ", len(trainingData))

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            np.savez_compressed(("compressed_data/{}").format(row["image"].split("/")[-1]), x=x_train, y=y_train)
        else:
            x_train = []
            y_train = []

            img = nib.load(trainingData[0]["image"]).get_fdata()
            label = nib.load(trainingData[0]["label"]).get_fdata()

            for i in range(img.shape[2]):
                if 1 in label[:,:,i]:
                    x_train.append(img[:,:,i])
                    y_train.append(label[:,:,i])

            counter += 1
            print("Reading slice: ", counter, " of: ", len(trainingData))

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            np.savez_compressed(("compressed_validation/{}").format(row["image"].split("/")[-1]), x=x_train, y=y_train)

convert_training_to_compressed_numpy()
#generate_validation_data()
