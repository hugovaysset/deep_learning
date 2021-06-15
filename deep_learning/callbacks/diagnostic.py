import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

plt.style.use('ggplot')

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
            plt.savefig(f"{self.save_dir}/epoch_{epoch}_{im_path}.png", bbox_inches="tight", format="png")


class SaveLearningCurvesCallbakc(tf.keras.callbacks.Callback):
    """
    Callback to plot and save the learning curves loss=f(epochs) and metrics=f(epochs) from epoch==0
    to epoch==current epoch. Allows to make a movie of the learning curves over time.
    """

    def __init__(self, save_dir, metrics=None):
        """
        save_dir (str): path to save the curves
        metrics (str or tf.keras.metrics): metrics to plot
        """
        self.save_dir = save_dir
        self.metrics = metrics

    def on_epoch_end(self, epoch, logs={}):
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
 
        # plot learning curves
        ax[0].plot(logs["loss"][:], "orange", label="loss")
        ax[0].plot(logs["val_loss"][:], "b", label="validation loss")
        ax[0].legend()
        ax[0].set_title("Training curves")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel(f"Loss")
        # ax[0].set_ylabel("Loss (MSE)")

        ax[1].plot(logs[f"{self.metrics}"][:], "orange", label=f"{self.metrics}")
        ax[1].plot(logs[f"val_{self.metrics}"][:], "b", label=f"validation {self.metrics}")
        ax[1].legend()
        ax[1].set_title("Training curves")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("IoU")

        fig.suptitle(f"Epoch {epoch}")

        plt.savefig(f"{self.save_dir}/epoch_{epoch}_learning_curve.png", format="png")
