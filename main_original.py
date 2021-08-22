import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from skimage.transform import resize

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def create_dataset(ftrain,idx_row,path_ds):
    N = len(idx_row)
    train_x = []
    train_y = []
    for i in range(N):
        idx = idx_row[i]
        imageName = os.path.join(path_ds,ftrain.iloc[idx][0])
        labName = os.path.join(path_ds,ftrain.iloc[idx][1])
        input_array = rgb2gray(mpimg.imread(imageName))
        input_array = resize(input_array, (input_array.shape[0] // 3, input_array.shape[1] // 3),
                       anti_aliasing=True)#downsample image so that its size is the same as the mask size
        train_x.append(input_array)
        mask = mpimg.imread(labName)
        if(i==0):
            print(input_array)
            print(mask)
            print(mask.shape)
        target_array = np.zeros((mask.shape[0],mask.shape[1],7))
        target_array[:,:,0]=np.where(mask == 1, 1, 0)
        target_array[:,:,1]=np.where(mask == 2, 1, 0)
        target_array[:,:,2]=np.where(mask == 3, 1, 0)
        target_array[:,:,3]=np.where(mask == 4, 1, 0)
        target_array[:,:,4]=np.where(mask == 5, 1, 0)
        target_array[:,:,5]=np.where(mask == 6, 1, 0)
        target_array[:,:,5]=np.where(mask == 7, 1, 0)
        train_y.append(target_array)
        if (i == 0):
            print(target_array)
    train_x = np.stack(train_x)
    train_x = np.expand_dims(train_x, axis=3)
    train_y = np.stack(train_y)
    print(train_x.shape)
    print(train_y.shape)
    return train_x,train_y

if __name__=="__main__":
    """ import data """
    path_ds = "tutorial/Tokaido_dataset"  # dir path
    ftrain = pd.read_csv(os.path.join(path_ds, 'files_train.csv'),header=None,index_col=None,delimiter=',')     # read csv (pandas)

    """ create training dataset and the corresponding label """
    col_valid = ftrain[5]   # boolean training
    idx_valid = [i for i in range(len(col_valid)) if col_valid[i]]      # training index
    train_x, train_y = create_dataset(ftrain, idx_valid[:100], path_ds)     # training data x,y
    validation_x, validation_y = create_dataset(ftrain, idx_valid[100:110], path_ds)    # validation data x,y
    test_x, test_y = create_dataset(ftrain, idx_valid[110:160], path_ds)    # test data x,y

    """ make train model (setting) """
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
    history = model.fit(train_x, train_y, batch_size=4, epochs=5,
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
