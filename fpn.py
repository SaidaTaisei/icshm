import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Model
from skimage.transform import resize,rescale
from PIL import Image
from Helper import Helper
import segmentation_models as sm

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
        width = 128
        height = 256
        print(i/N)
        file_name = index_array[i,0]
        imageName = os.path.join(path_ds,"image/"+file_name)
        labName = os.path.join(path_ds,"label/component/"+file_name)  # If it is not a component, rewrite it here.
        input_array = mpimg.imread(imageName)
        #input_array = rgb2gray(input_array)     # converting it to a gray image, but it's just a tutorial and it's not necessary.
        input_array = resize(input_array, (width, height), anti_aliasing=True)
        """
        The tutorial is downsampled, but you don't need it because Project 2 has the same resolution.
        input_array = resize(input_array, (input_array.shape[0] // 3, input_array.shape[1] // 3),
                       anti_aliasing=True)#downsample image so that its size is the same as the mask size
        """
        train_x.append(input_array)
        mask = np.asarray(Image.open(labName).convert('RGB'))   # import label image (The method is different from the above because it is read as an int type)
        mask = (255*resize(mask, (width, height,mask.shape[2]), anti_aliasing=False)).astype(np.int32)
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
    #train_x = np.expand_dims(train_x, axis=3)
    train_y = np.stack(train_y)
    print(train_x.shape)
    print(train_y.shape)
    return train_x,train_y


if __name__=="__main__":
    # tf.test.is_built_with_cuda()
    # print(len(tf.config.experimental.list_physical_devices('GPU')))

    """ import data """
    path_ds = "QuakeCity"  # dir path
    ftrain = pd.read_csv(os.path.join(path_ds, 'train.csv'),header=None,index_col=None,delimiter=',')     # read csv (pandas)
    train_index_array = ftrain.values


    """ create training dataset and the corresponding label """
    """ As with the original tutorial, divide the training dataset into three parts """
    """ Since the test data cannot be verified this time, we will divide it into three parts from the training data """
    """ Slices in the array are a bit different from matlab, so be careful if you're a matlab sect """
    """ Please correct the position of the slice as appropriate. """
    isCreate = False
    train_x_path = "train_x_array"
    train_y_path = "train_y_array"
    validation_x_path = "validation_x_array"
    validation_y_path = "validation_y_array"
    test_x_path = "test_x_array"
    test_y_path = "test_y_array"
    train_x, train_y = 0, 0  # training data x,y
    validation_x, validation_y = 0, 0  # validation data x,y
    test_x, test_y = 0, 0  # test data x,y
    if(os.path.exists(train_x_path+".npy") and (not isCreate)):
        train_x, train_y = np.load(train_x_path+".npy"),np.load(train_y_path+".npy")  # training data x,y
        validation_x, validation_y = np.load(validation_x_path+".npy"),np.load(validation_y_path+".npy")  # validation data x,y
        test_x, test_y = np.load(test_x_path+".npy"),np.load(test_y_path+".npy")  # test data x,y
    else:
        train_x, train_y = create_dataset(train_index_array[0:800,:], path_ds)     # training data x,y
        validation_x, validation_y = create_dataset(train_index_array[800:900,:], path_ds)    # validation data x,y
        test_x, test_y = create_dataset(train_index_array[900:1000,:], path_ds)    # test data x,y
        np.save(train_x_path + ".npy",train_x)
        np.save(train_y_path + ".npy",train_y)  # training data x,y
        np.save(validation_x_path + ".npy",validation_x)
        np.save(validation_y_path + ".npy",validation_y)  # validation data x,y
        np.save(test_x_path + ".npy",test_x)
        np.save(test_y_path + ".npy",test_y)  # test data x,y

    """ make train model (setting) """
    """ If you have a GPU, consider using it. """
    """ Depending on the case, you can expect about 10 times faster speed. """
    tf.keras.backend.clear_session()
    Input_CHANNELS = 1
    IMG_HEIGHT = train_y.shape[1]
    IMG_WIDTH = train_y.shape[2]

    """ create unet model """

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, Input_CHANNELS))
    BACKBONE = 'efficientnetb5'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    # define model
    model = sm.FPN(BACKBONE, encoder_weights=None,input_shape=(None, None, 3),classes=train_y.shape[-1])
    #tf.keras.utils.plot_model(model, show_shapes=True)
    #plt.show()
    print(model.summary())

    model.compile(optimizer='adam',
                  loss=sm.losses.bce_jaccard_loss,
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.MeanIoU(train_y.shape[-1])])

    """ train here, but it's going to take some time, so don't rush """
    history = model.fit(train_x, train_y, batch_size=8, epochs=100,
                        validation_data=(validation_x, validation_y))
    print(history)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #tf.keras.utils.plot_model(model, show_shapes=True)
    #plt.show()
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
    predict_y = model.predict(test_x)
    print(predict_y.shape)
    Helper.saveImage(test_x,test_y,predict_y)
