import numpy as np
import imageio
import os
import sys
import glob
import cv2
import skimage.transform
import re

# Utils for importing and exporting images from directory, or volumes, etc.

# Preprocessing

def alphanumeric_sort( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def normalize_image(image):
    """
    Channel-wise normalization
    """
    if image.min() != image.max():
        return (image - image.min(axis=(0, 1))) / (image.max(axis=(0, 1)) - image.min(axis=(0, 1)))
    else:
        print("Blank image.")
        return (image + 1e-5) / (image.max(axis=(0, 1)) + 1e-5)

def normalize_batch(images):  #TODO: vectoriser cette fonction pour aller plus vite
    normalized = np.zeros(images.shape)
    for i, im in enumerate(images):
        normalized[i] = normalize_image(im)
    assert normalized.shape == images.shape
    assert normalized.min() == 0. and normalized.max() == 1.0
    return normalized

def reshape(image, target_shape, interpolation=1):
    """
    Reshape image to target shape using skimage
    """
    return skimage.transform.resize(image, target_shape, order=interpolation)

def reshape_batch(images, target_shape):
    if images.ndim == 4:
        reshaped = np.zeros((images.shape[0], target_shape[0], target_shape[1], images.shape[-1]))
    elif images.ndim == 3:
        reshaped = np.zeros((images.shape[0], target_shape[0], target_shape[1]))
    for i, im in enumerate(images):
        reshaped[i] = reshape(im, target_shape)
    return reshaped

def add_channel(image, axis=-1):
    return np.expand_dims(image, axis=axis)

def preprocess(images, normalize=True, add_axis=None, target_shape=(512, 512)):
    preprocessed = images

    if images.ndim == 4: # assuming batch
        print(f"Processing a batch of dimension: {images.shape}")
        if target_shape[0] != images.shape[1] and target_shape[1] != images.shape[2]:
            preprocessed = reshape_batch(preprocessed, target_shape)
        if normalize:
            preprocessed = normalize_batch(preprocessed)
        if add_axis and add_axis is not None:
            preprocessed = np.expand_dims(preprocessed, axis=add_axis)
    
    elif images.ndim == 3 and images.shape[-1] > 10: # more than 10 on the last axis -> assuming batch
        print(f"Processing a batch of dimension: {images.shape}")
        if target_shape[0] != images.shape[1] and target_shape[1] != images.shape[2]:
            preprocessed = reshape_batch(preprocessed, target_shape)
        if normalize:
            preprocessed = normalize_batch(preprocessed)
        if add_axis and add_axis is not None:
            preprocessed = np.expand_dims(preprocessed, axis=add_axis)

    elif images.ndim == 3 and images.shape[-1] <= 10: # less than 10 on the last axis -> assuming image with channel
        print(f"Processing an image of dimension: {images.shape}")
        if target_shape[0] != images.shape[0] and target_shape[1] != images.shape[1]:
            preprocessed = reshape(preprocessed, target_shape)
        if normalize:
            preprocessed = normalize_image(preprocessed)
        if add_axis and add_axis is not None:
            preprocessed = np.expand_dims(preprocessed, axis=add_axis)


    elif images.ndim == 2: # assuming batch
        print(f"Processing an image of dimension: {images.shape}")
        if target_shape[0] != images.shape[0] and target_shape[1] != images.shape[1]:
            preprocessed = reshape(preprocessed, target_shape)
        if normalize:
            preprocessed = normalize_image(preprocessed)
        if add_axis and add_axis is not None:
            preprocessed = np.expand_dims(preprocessed, axis=add_axis)

    else:
        print("Unknown image dimension.")

    return preprocessed

# Input/Output

def is_batch(images):
    return images.ndim == 4 or (images.ndim == 3 and images.shape[-1] > 10)

def is_image(image):
    return images.ndim == 2 or (images.ndim == 3 and images.shape[-1] <= 10)

def to_batch_dim(image):
    """
    Add the batch and channel axis to go from image shape (x, y) or (x, y, c) to batch shape.
    """
    if is_image(image) and image.ndim == 2:
        return np.expand_dims(np.expand_dims(image, axis=-1), axis=0)
    elif is_image(image) and image.ndim == 3:
        return np.expand_dims(image, axis=0)
    else:
        print("Already in batch shape ?")
        return image

def imread(path):
    return imageio.imread(path)

def read_from_directory(path, restrict_to=-1):
    X = []
    max_id = len(os.listdir(path)) if restrict_to == -1 else restrict_to
    for i, im in zip(range(max_id), alphanumeric_sort(os.listdir(path))):
        X.append(imageio.imread(os.path.join(path, im)))
    return np.array(X)

def read_from_stack(path):
    return imageio.volread(path)

def merge_channels(channels): 
    for i, chan in enumerate(channels):
        if not is_batch(chan):  # not batch, i.e. image
            channels[i] = to_batch_dim(chan)
        elif is_batch(chan) and chan.ndim == 3:  # batch but no channel axis
            channels[i] = np.expand_dims(chan, axis=-1)
    return np.concatenate([chan for chan in channels], axis=-1)