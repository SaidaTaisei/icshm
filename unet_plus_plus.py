import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Model
from skimage.transform import resize,rescale
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tfdeterminism import patch


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
        input_array = rgb2gray(input_array)     # converting it to a gray image, but it's just a tutorial and it's not necessary.
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
    train_x = np.expand_dims(train_x, axis=3)
    train_y = np.stack(train_y)
    print(train_x.shape)
    print(train_y.shape)
    return train_x,train_y

def conv_batchnorm_relu_block(input_tensor, nb_filter, kernel_size=3):

    x = layers.Conv2D(nb_filter, (kernel_size, kernel_size), padding='same')(input_tensor)
    x = layers.BatchNormalization(axis=2)(x)
    x = layers.Activation('relu')(x)

    return x


def model_build_func(input_shape, n_labels, using_deep_supervision=False):

    nb_filter = [32,64,128,256,512,1024]

    # Set image data format to channels first
    global bn_axis

    K.set_image_data_format("channels_last")
    bn_axis = -1
    inputs = Input(shape=input_shape, name='input_image')

    conv1_1 = conv_batchnorm_relu_block(inputs, nb_filter=nb_filter[0])
    pool1 = layers.AvgPool2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = conv_batchnorm_relu_block(pool1, nb_filter=nb_filter[1])
    pool2 = layers.AvgPool2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = layers.concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = conv_batchnorm_relu_block(conv1_2,  nb_filter=nb_filter[0])

    conv3_1 = conv_batchnorm_relu_block(pool2, nb_filter=nb_filter[2])
    pool3 = layers.AvgPool2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = layers.concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = conv_batchnorm_relu_block(conv2_2, nb_filter=nb_filter[1])

    up1_3 = layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = layers.concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = conv_batchnorm_relu_block(conv1_3, nb_filter=nb_filter[0])

    conv4_1 = conv_batchnorm_relu_block(pool3, nb_filter=nb_filter[3])
    pool4 = layers.AvgPool2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = layers.concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = conv_batchnorm_relu_block(conv3_2, nb_filter=nb_filter[2])

    up2_3 = layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = layers.concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = conv_batchnorm_relu_block(conv2_3, nb_filter=nb_filter[1])

    up1_4 = layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = layers.concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = conv_batchnorm_relu_block(conv1_4, nb_filter=nb_filter[0])

    conv5_1 = conv_batchnorm_relu_block(pool4, nb_filter=nb_filter[4])

    up4_2 = layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = layers.concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = conv_batchnorm_relu_block(conv4_2, nb_filter=nb_filter[3])

    up3_3 = layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = layers.concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = conv_batchnorm_relu_block(conv3_3, nb_filter=nb_filter[2])

    up2_4 = layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = layers.concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = conv_batchnorm_relu_block(conv2_4, nb_filter=nb_filter[1])

    up1_5 = layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = layers.concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = conv_batchnorm_relu_block(conv1_5, nb_filter=nb_filter[0])

    nestnet_output_1 = layers.Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_1',padding='same')(conv1_2)
    nestnet_output_2 = layers.Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_2', padding='same' )(conv1_3)
    nestnet_output_3 = layers.Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_3', padding='same')(conv1_4)
    nestnet_output_4 = layers.Conv2D(n_labels, (1, 1), activation='sigmoid', name='output_4', padding='same')(conv1_5)

    if using_deep_supervision:
        model = Model(inputs=inputs, outputs=[nestnet_output_1,
                                            nestnet_output_2,
                                            nestnet_output_3,
                                            nestnet_output_4])
    else:
        model = Model(inputs=inputs, outputs=nestnet_output_4)

    return model

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

    print(train_x.shape)
    print(train_y.shape)

    """ make train model (setting) """
    """ If you have a GPU, consider using it. """
    """ Depending on the case, you can expect about 10 times faster speed. """
    tf.keras.backend.clear_session()
    Input_CHANNELS = 1
    IMG_HEIGHT = train_y.shape[1]
    IMG_WIDTH = train_y.shape[2]

    """ create unet model """
    input_shape = (IMG_HEIGHT, IMG_WIDTH, Input_CHANNELS)
    model = model_build_func(input_shape=input_shape,n_labels=train_y.shape[-1],using_deep_supervision=False)

    tf.keras.utils.plot_model(model, show_shapes=True)
    plt.show()
    print(model.summary())

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.MeanIoU(train_y.shape[-1])])

    """ train here, but it's going to take some time, so don't rush """
    history = model.fit(train_x, train_y, batch_size=8, epochs=300,
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