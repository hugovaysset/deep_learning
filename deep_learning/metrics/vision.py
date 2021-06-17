import tensorflow as tf
import tensorflow.keras.backend as K

def IoU_metric(y_true, y_pred):
    """
    Computes the Intersection over Union metrics for a tensor of shape
    (batch, height, width, channels). The IoU is first computed for each
    image (y_true, y_pred) after y_pred has been thresholded to get a
    tensor composed of only ones and zeros. Finally the mean IoU is taken
    across all images of the batch mean(axis=0).
    """
    threshold = tf.constant(0.5, dtype=tf.float32)
    y_pred = K.cast(tf.math.greater(y_pred, threshold), dtype="float32")
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3))
    union = K.sum(y_true + y_pred, axis=(1, 2, 3)) - intersection
    return K.mean((intersection + K.epsilon()) / (union + K.epsilon()), axis=0)
