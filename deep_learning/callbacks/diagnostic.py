import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from shutil import move, rmtree
import tensorflow as tf
from tensorflow import keras

plt.style.use('ggplot')

class PredictOnImagesCallback(tf.keras.callbacks.Callback):
    """
    Callback to make segmentation predictions on test images in the end of each epoch.
    """

    def __init__(self, generator, save_dir, model_name, save_freq=1, n_images_to_predict=1):
        """
        generator (Generator): image generator giving tuples of (image, mask) on the fly
        save_dir (str): save path for the predictions
        model_name (str): name of the model (to place the save folder inside it in the end)
        save_freq (int): save predictions every save_freq epoch
        n_images_to_predict (int): number of images on which to make predictions
        """
        self.generator = generator
        self.save_dir = save_dir
        self.model_name = model_name
        self.n_images_to_predict = n_images_to_predict
        self.binarize = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        self.save_freq = save_freq
        
        if os.path.isdir(self.save_dir):
            rmtree(self.save_dir)
        os.mkdir(self.save_dir)
    
    def IoU(self, y_true, y_pred):
        threshold = tf.constant(0.5, dtype=tf.float32)
        y_pred = K.cast(tf.math.greater(y_pred, threshold), dtype="float32")
        intersection = K.sum(y_true * y_pred, axis=(1, 2, 3))
        union = K.sum(y_true + y_pred, axis=(1, 2, 3)) - intersection
        return K.mean((intersection + K.epsilon()) / (union + K.epsilon()), axis=0)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.save_freq == 0:
            images = np.concatenate([self.generator[k][0] for k in range(self.n_images_to_predict // self.generator.batch_size)], axis=0)

            masks = np.concatenate([self.generator[k][1] for k in range(self.n_images_to_predict // self.generator.batch_size)], axis=0) > 0.5
            predictions = self.model.predict(images) > 0.5  
            metrics = self.IoU(masks, predictions).numpy()

            for i, (im, pred) in enumerate(zip(images, predictions)):
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                for k in range(self.generator.n_channels_ims):
                    if self.generator.n_channels_ims == 1 and k == 0:
                        ax.imshow(im[:, :, k], cmap="gray")
                    elif self.generator.n_channels_ims > 1 and k == 0:
                        ax.imshow(im[:, :, k], cmap="Reds")
                    else:
                        ax.imshow(im[:, :, k], cmap="gray", alpha=0.3)
                ax.imshow(pred.squeeze(-1), cmap="Blues", alpha=0.4)
                ax.set_title(f"Epoch {epoch+1}, IoU = {round(metrics, 2)}")
                plt.axis('off')
                plt.savefig(f"{self.save_dir}/epoch_{epoch+1}_im_{i}.png", bbox_inches="tight", format="png")
                plt.close(fig)

    def on_train_end(self, logs=None):
        if os.path.isdir(self.model_name):
            rmtree(self.model_name)
        if not os.path.isdir(self.model_name):
            os.mkdir(self.model_name)
        move(self.save_dir, self.model_name)


class PlotLearningCurvesCallback(tf.keras.callbacks.Callback):
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


class PlotLossHistogramCallbakc(tf.keras.callbacks.Callback):
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
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
 
        # plot learning curves
        ax[0].plot(logs["loss"][:], "orange", label="loss")
        ax[0].plot(logs["val_loss"][:], "b", label="validation loss")
        ax[0].legend()
        ax[0].set_title("Training curves")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel(f"Loss")
        # ax[0].set_ylabel("Loss (MSE)")

        fig.suptitle(f"Epoch {epoch}")

        plt.savefig(f"{self.save_dir}/epoch_{epoch}_learning_curve.png", format="png")
