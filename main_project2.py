import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from skimage.transform import resize,rescale
from PIL import Image

def rgb2gray(rgb):
    """
    Conversion from rgb to gray color
    not sure if this number is good
    :param rgb: numpy_array rgb color matrix
    :return: numpy_array gray color matrix
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def create_dataset(index_array, path_ds):
    """
    For the time being, it is for components, so please rewrite as appropriate.
    Project 2 is all png images, so basically you only need to rewrite the directory path.
    :param index_array: numpy_array
    :param path_ds: directory path
    :return: x,y dataset
    """
    N = index_array.shape[0]
    train_x = []
    train_y = []
    for i in range(N):
        print(i/N)
        file_name = index_array[i,0]
        imageName = os.path.join(path_ds,"image/"+file_name)
        labName = os.path.join(path_ds,"label/component/"+file_name)  # If it is not a component, rewrite it here.
        input_array = rgb2gray(mpimg.imread(imageName))     # converting it to a gray image, but it's just a tutorial and it's not necessary.
        input_array = resize(input_array, (input_array.shape[0] // 8, input_array.shape[1] // 8),
                             anti_aliasing=True)
        """
        The tutorial is downsampled, but you don't need it because Project 2 has the same resolution.
        input_array = resize(input_array, (input_array.shape[0] // 3, input_array.shape[1] // 3),
                       anti_aliasing=True)#downsample image so that its size is the same as the mask size
        """
        train_x.append(input_array)
        mask = np.asarray(Image.open(labName).convert('RGB'))   # import label image (The method is different from the above because it is read as an int type)
        mask = (255*resize(mask, (mask.shape[0]//8,mask.shape[1]//8,mask.shape[2]), anti_aliasing=False)).astype(np.int32)
        #print(mask)
        """ Since there were 3 dimensions, it is judged by multiplying any, but I think there is a smarter method. """
        """ It takes longer than the tutorial due to the increased dimensions and the increased resolution. """
        target_array = np.zeros((mask.shape[0],mask.shape[1],7))
        target_array[:, :, 0] = (np.any(mask == 202,axis=-1) * np.any(mask == 150,axis=-1) * np.any(mask == 150,axis=-1)).astype(np.int32)   # wall - (202,150,150)
        target_array[:, :, 1] = (np.any(mask == 198,axis=-1) * np.any(mask == 186,axis=-1) * np.any(mask == 100,axis=-1)).astype(np.int32)   # beam - (198,186,100)
        target_array[:, :, 2] = (np.any(mask == 167,axis=-1) * np.any(mask == 183,axis=-1) * np.any(mask == 186,axis=-1)).astype(np.int32)   # column - (167,183,186)
        target_array[:, :, 3] = (np.any(mask == 255,axis=-1) * np.any(mask == 255,axis=-1) * np.any(mask == 133,axis=-1)).astype(np.int32)   # window frame - (255,255,133)
        target_array[:, :, 4] = (np.any(mask == 192,axis=-1) * np.any(mask == 192,axis=-1) * np.any(mask == 206,axis=-1)).astype(np.int32)   # window pane - (192,192,206)
        target_array[:, :, 5] = (np.any(mask == 32,axis=-1) * np.any(mask == 80,axis=-1) * np.any(mask == 160,axis=-1)).astype(np.int32)   # balcony - (32,80,160)
        target_array[:, :, 6] = (np.any(mask == 193,axis=-1) * np.any(mask == 134,axis=-1) * np.any(mask == 1,axis=-1)).astype(np.int32)   # slab - (193,134,1)
        if (i == 0):
            print(input_array)
            print(target_array)
        train_y.append(target_array)
    train_x = np.stack(train_x)
    train_x = np.expand_dims(train_x, axis=3)
    train_y = np.stack(train_y)
    print(train_x.shape)
    print(train_y.shape)
    return train_x,train_y


if __name__=="__main__":
    """ import data """
    path_ds = "QuakeCity"  # dir path
    ftrain = pd.read_csv(os.path.join(path_ds, 'train.csv'),header=None,index_col=None,delimiter=',')     # read csv (pandas)
    train_index_array = ftrain.values


    """ create training dataset and the corresponding label """
    """ As with the original tutorial, divide the training dataset into three parts """
    """ Since the test data cannot be verified this time, we will divide it into three parts from the training data """
    """ Slices in the array are a bit different from matlab, so be careful if you're a matlab sect """
    """ Please correct the position of the slice as appropriate. """
    train_x, train_y = create_dataset(train_index_array[0:800,:], path_ds)     # training data x,y
    validation_x, validation_y = create_dataset(train_index_array[800:900,:], path_ds)    # validation data x,y
    test_x, test_y = create_dataset(train_index_array[900:1000,:], path_ds)    # test data x,y

    """ make train model (setting) """
    """ If you have a GPU, consider using it. """
    """ Depending on the case, you can expect about 10 times faster speed. """
    tf.keras.backend.clear_session()
    model = models.Sequential()     # create empty model
    # add layer
    model.add(
        layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=train_x.shape[1:], padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=train_y.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))
    print(model.summary())

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.MeanIoU(train_y.shape[-1])])

    """ train here, but it's going to take some time, so don't rush """
    history = model.fit(train_x, train_y, batch_size=50, epochs=5,
                        validation_data=(validation_x, validation_y))
    print(history)
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.plot(history.epoch, loss, 'r', label='Training loss')
    plt.plot(history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()

    """ Evaluate the model on the test data using evaluate """
    print("Evaluate on test data")
    results = model.evaluate(test_x, test_y, batch_size=1)
    print("test results:", results)

    """ Predict Test """
    predict = model.predict(test_x)