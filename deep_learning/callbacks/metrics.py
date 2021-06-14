import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

class AUCCallback(tf.keras.callbacks.Callback):
    """
    Callback used for any model that contains a "is_anormal" method (e.g. A VAE predicts images from images. The
    associated is_anormal method predicts labels (0 for normal, 1 for anormal) from those images and of course ground
    truth images.
    - After each training epoch: computes AUC score to measure the model performance
    - In the end of the training step : plots the ROC curve of the model (to do: store the ROC curve every two or three
    steps and plot them all in the end)
    """

    def __init__(self, x, y_true, threshold=0.5, prefix="val"):
        """
        :param gt_images: reference images (validation set)
        :param true_labels: array containing the true labels binarized (i.e. 0 for normal class and 1 for all other
        class which is considered as abnormal)
        """
        self.x = x
        self.y_true = y_true
        self.threshold = pix_threshold
        self.im_threshold = im_threshold
        self.prefix = prefix

        self.binarize = np.vectorize(lambda x: 1 if x > threshold else 0)

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.binarize(self.model.predict(self.gt_images))
        auc = tf.keras.metrics.AUC()
        auc.update_state(self.y_true, y_pred)
        res = auc.result()
        logs[f"{self.prefix}_AUC"] = res
        print(f"\nAUC = {res}")