import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

class PredictOnImagesCallback(tf.keras.callbacks.Callback):
    """
    Callback to make segmentation predictions on test images in the end of each epoch.
    """

    def __init__(self, image_dir, save_dir, n_images_to_predict=10):
        """
        image_dir (str): path to images
        save_dir (str): save path for the predictions
        """
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.n_images_to_predict = n_images_to_predict

    def on_epoch_end(self, epoch, logs={}):
        batch = np.random.choice(os.listdir(self.image_dir), size=self.n_images_to_predict, replace=False)
        for im_path in batch:
            im = imageio.imread(f"{self.image_dir}/{im_path}")
            pred = self.model.predict(np.expand_dims(im, axis=-1))  # add channel axis
            # imageio.imwrite(f"{self.save_dir}/epoch_{epoch}_{im_path}")
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(im.squeeze(-1), cmap="gray")
            ax.imshow(pred.squeeze(-1), cmap="Blues", alpha=0.4)
            plt.axis('off')
            plt.savefig(f"{self.save_dir}/epoch_{epoch}_{im_path}", bbox_inches="tight", format="png")
