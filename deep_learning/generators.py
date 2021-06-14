%matplotlib inline

import cv2
import imageio
import numpy as np
import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, layers, models

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import Iterator, ImageDataGenerator

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import Iterator, ImageDataGenerator
import tensorflow.keras.backend as K

import skimage.transform

class ImageMaskGenerator(Sequence):
    """
    Generates images and masks for performing data augmentation in Keras (e.g. to train an image segmentation network).
    We inherit from Sequence (instead of directly using the keras ImageDataGenerator) since we want to perform augmentation 
    on both the input image AND the mask (target). This mechanism needs to be implemented in this class. This class also allows 
    to implement new augmentation transforms that are not implemented in the core Keras class (illumination, etc.).
    See : https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    and https://stackoverflow.com/questions/56758592/how-to-customize-imagedatagenerator-in-order-to-modify-the-target-variable-value
    for more details.
    """

    def __init__(self, X_set, Y_set=None, # input images and masks
                 batch_size: int=32, dim: tuple=(512, 512),
                 n_channels_ims: int=1, n_channels_masks: int=1, # informations 
                 shuffle: bool=True, normalize_ims=True, normalize_masks=False, reshape=False, crop=None,# preprocessing params
                 **kwargs): # data augmentation params
        """
        X_set (list, array or str): pointer to the images (Bright-Field). If str
        the string is assumed to be pointing at some directory.
        Y_set (list; array or str): pointer to the masks (target). If str
        the string is assumed to be pointing at some directory.
        batch_size (int): size of the batch
        dim (tuple): dimension of the images
        n_channels_ims (int) : number of channels of the images (1 for TIF)
        shuffle (bool): Shuffle the dataset between each training epoch
        crop (tuple): Target dim of one image after cropping
        normalize (bool): normalize the images and masks in the beginning
        reshape (bool): reshape the images and masks to (dim, dim, n_channels_ims)
        histogram_equalization (bool): perform histogram equalization to improve
        rendering using opencv
        horiz_flip_percent ()
        vert_flip_percent
        """
        # super().__init__(n, batch_size, shuffle, seed)
        self.dim = dim
        self.im_size = dim
        self.batch_size = batch_size
        self.n_channels_ims = n_channels_ims
        self.n_channels_masks = n_channels_masks
        
        # build the X_set in an array. If X_set is a directory containing images
        # then self.X_set doesn't contains the images but the file names, but it
        # is transparent for the user.
        if type(X_set) == list:
            self.from_directory_X = False
            self.X_set = np.array(X_set)
        elif type(X_set) == np.array:
            self.from_directory_X = False
            self.X_set = X_set
        elif type(X_set) == str: # assuming a path
            self.from_directory_X = True
            self.X_dir = X_set # path to the images dir
#             if self.n_channels_ims == 1:
#                 self.X_set = np.array(sorted(os.listdir(X_set))) # sorted guarantees the order
#             else: # n_channels_ims > 1 : several channels per image
            self.X_set = []
            for k in range(0, len(os.listdir(X_set)), self.n_channels_ims):
                self.X_set.append(np.array(os.listdir(X_set)[k:k+self.n_channels_ims]))
            self.X_set = np.array(self.X_set)
        else:
            raise TypeError("X_set should be list, array or path")
        
        # build the Y_set in an array
        if type(Y_set) == list:
            self.from_directory_Y = False
            self.Y_set = np.array(Y_set)
        elif type(Y_set) == np.array:
            self.from_directory_Y = False
            self.Y_set = Y_set
        elif type(Y_set) == str: # assuming a path
            self.from_directory_Y = True
            self.Y_dir = Y_set # path to the masks dir
            self.Y_set = []
            for k in range(0, len(os.listdir(Y_set)), self.n_channels_masks):
                self.Y_set.append(np.array(os.listdir(Y_set)[k:k+self.n_channels_masks]))
            self.Y_set = np.array(self.Y_set)
        else:
            raise TypeError("Y_set should be list, array or path")

        # Check if there are the same number of images in X (images) and Y (masks)
        assert self.X_set.shape[0] != 0 and self.Y_set.shape[0] != 0, print(f"Directory '{X_set}' is empty!")
        assert self.X_set.shape[0] == self.Y_set.shape[0], print(f"{self.X_set.shape[0]} images != {self.Y_set.shape[0]} masks")

        self.shuffle = shuffle

        # Preprocessing parameters
        self.normalize_ims = normalize_ims
        self.normalize_masks = normalize_masks
        self.reshape = reshape
        self.crop = crop

        # The Keras generator that will be used to perform data augmentation 
        self.generator = ImageDataGenerator(**kwargs)

        # Initialize the indices (shuffle if asked)
        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Number of batches per epoch : we evenly split the train set into samples
        of size batch_size.
        """
        return int(np.floor(self.X_set.shape[0] / self.batch_size))
        
    def __getitem__(self, index: int):
        """
        Generate one batch of data.
        """
        if index >= self.__len__():
            raise IndexError
        
        # Generate indices corresponding to the images in the batch
        indices = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate the batch
        X, Y = self.__data_generation(indices)
        return X, Y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch. self.indexes is used to retrieve the
        samples and organize them into batches.
        If shuffle : randomizes the order of the samples in order to give 
        different training batches at each epoch.
        """
        self.indexes = np.arange(self.X_set.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs):
        """
        Generates data containing batch_size samples. This is where we load the
        images if they are in a directory, and apply transformations to them.
        list_IDs [int]: list of the image IDs
        """ 
        # Load data (from directory or from X_set depending on the given data)
        if self.from_directory_X:
            batch_X = []
            for im in list_IDs:
                channels = []
                for k in range(self.n_channels_ims):
                    channels.append(np.expand_dims(imageio.imread(f"{self.X_dir}/{self.X_set[im, k]}"), axis=-1)) # add channel axis
                batch_X.append(np.concatenate(channels, axis=-1))
            batch_X = np.array(batch_X)
        else:
            batch_X = self.X_set[list_IDs]

        if self.from_directory_Y:
            batch_Y = []
            for im in list_IDs:
                channels = []
                for k in range(self.n_channels_masks):
                    channels.append(np.expand_dims(imageio.imread(f"{self.Y_dir}/{self.Y_set[im, k]}"), axis=-1)) # add channel axis
                batch_Y.append(np.concatenate(channels, axis=-1))
            batch_Y = np.array(batch_Y) 
        else:
            batch_Y = self.Y_set[list_IDs]

        # Preprocessing
        if self.crop is not None:
            batch_X = self.perf_crop(batch_X)
            batch_Y = self.perf_crop(batch_Y)

        if self.reshape:
            batch_X = self.perf_reshape(batch_X, is_images=True)
            batch_Y = self.perf_reshape(batch_Y, is_images=False)

        if self.normalize_ims:
            batch_X = self.perf_normalize(batch_X)
        if self.normalize_masks:
            batch_Y = self.perf_normalize(batch_Y)

#         if self.n_channels_ims == 3:
#             batch_X = np.concatenate([batch_X, batch_X, batch_X], axis=-1)

        # Perform the SAME transformation on the image and on the mask
        for i, (img, mask) in enumerate(zip(batch_X, batch_Y)):
            transform_params = self.generator.get_random_transform(img.shape)
            batch_X[i] = self.generator.apply_transform(img, transform_params)
            batch_Y[i] = self.generator.apply_transform(mask, transform_params)
            
        return batch_X, batch_Y        

    # Preprocessing functions
    def perf_crop(self, images):
        crop_X = int((images.shape[1] - self.crop[0]) // 2)
        crop_Y = int((images.shape[2] - self.crop[1]) // 2)
        assert (crop_X >= 0 and crop_Y >= 0), print(f"Target size after cropping {self.crop} should be lower than the initial shape {(images.shape[1], images.shape[2])}.")
        new_batch = np.empty((self.batch_size, *self.crop, images.shape[3]))
        for i, img in enumerate(images):
            if crop_X != 0 and crop_Y != 0:
                new_batch[i] = img[crop_X:-crop_X, crop_Y:-crop_Y]
            elif crop_X != 0:
                new_batch[i] = img[crop_X:-crop_X, :]
            elif crop_Y != 0:
                new_batch[i] = img[:, crop_Y:-crop_Y]
            else:
                new_batch[i] = img
        return new_batch

    def perf_reshape(self, images, is_images=True):
        """
        images (np.array): batch of images of shape (batch_size, n_rows, n_cols, n_chans)
        is_images (bool): is it a batch of images (True) or masks (False)
        """
        if is_images:  # batch of images
            new_batch = np.empty((self.batch_size, *self.im_size, self.n_channels_ims))
            for i, img in enumerate(images): # the resize function normalizes the images anyways...
                new_batch[i] = skimage.transform.resize(img, (*self.im_size, self.n_channels_ims), anti_aliasing=True)
        else:  # batch of masks
            new_batch = np.empty((self.batch_size, *self.im_size, self.n_channels_masks))
            for i, img in enumerate(images):
                new_batch[i] = skimage.transform.resize(img, (*self.im_size, self.n_channels_masks), anti_aliasing=True)
        return new_batch

    def perf_normalize(self, images):
        """
        Performs per image, per channel normalization by substracting the min and dividing by (max - min)
        """
        new_batch = np.empty(images.shape)
        for i, img in enumerate(images):
            assert (np.min(img, axis=(0, 1)) != np.max(img, axis=(0, 1))).all(), print("Cannot normalize an image containing only 0 or 1 valued pixels. There is likely an empty image in the training set.\nIf cropping was used,"
                                                                                       "maybe the mask doesn't contain any white pixel in the specific region.")
            new_batch[i] = (img - np.min(img, axis=(0, 1))) / (np.max(img, axis=(0, 1)) - np.min(img, axis=(0, 1)))
        return new_batch
