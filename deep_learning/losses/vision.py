import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow.keras.backend as K


def jaccard_distance(smooth=20):

    def jaccard_distance_fixed(y_true, y_pred):
        """
        Calculates mean of Jaccard distance as a loss function. The input tensors are 
        assumed to be of shape (batch, height, width, channels). The Jaccard loss is
        first computed for each image (axis=1, 2, 3) and then averaged across images
        of the batch (axis=0).
        """ 
        intersection = K.sum(K.abs(y_true * y_pred), axis=(1, 2, 3))
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=(1, 2, 3))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return K.mean((1 - jac) * smooth, axis=0)
    
    return jaccard_distance_fixed


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
    model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss
    return binary_focal_loss_fixed


def yolo_loss(anchors=np.array([[3.0, 1.5], [2.0, 2.0], [1.5, 3.0]]), cell_shape=(16, 16), lambda_coord=5, lambda_noobj=0.5, iou_threshold=0.5):

    def yolo_loss_fixed(labels, preds):
        cellule_x, cellule_y = cell_shape[0], cell_shape[1]
        nbr_boxes = len(anchors)
        seuil_iou_loss = iou_threshold

        grid=tf.meshgrid(tf.range(cellule_x, dtype=tf.float32), tf.range(cellule_y, dtype=tf.float32))
        grid=tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        grid=tf.tile(grid, (1, 1, nbr_boxes, 1))
        
        preds_xy = tf.math.sigmoid(preds[:, :, :, :, 0:2])+grid
        preds_wh = preds[:, :, :, :, 2:4]
        preds_conf = tf.math.sigmoid(preds[:, :, :, :, 4])
        preds_classe = tf.math.sigmoid(preds[:, :, :, :, 5:])

        preds_wh_half=preds_wh/2
        preds_xymin=preds_xy-preds_wh_half
        preds_xymax=preds_xy+preds_wh_half
        preds_areas=preds_wh[:, :, :, :, 0]*preds_wh[:, :, :, :, 1]

        l2_xy_min=labels2[:, :, 0:2]
        l2_xy_max=labels2[:, :, 2:4]
        l2_area  =labels2[:, :, 4]
        
        preds_xymin=tf.expand_dims(preds_xymin, 4)
        preds_xymax=tf.expand_dims(preds_xymax, 4)
        preds_areas=tf.expand_dims(preds_areas, 4)

        labels_xy    =labels[:, :, :, :, 0:2]
        labels_wh    =tf.math.log(labels[:, :, :, :, 2:4]/anchors)
        labels_wh=tf.where(tf.math.is_inf(labels_wh), tf.zeros_like(labels_wh), labels_wh)
        
        conf_mask_obj=labels[:, :, :, :, 4]
        labels_classe=labels[:, :, :, :, 5:]
        
        conf_mask_noobj=[]
        for i in range(len(preds)):
            xy_min=tf.maximum(preds_xymin[i], l2_xy_min[i])
            xy_max=tf.minimum(preds_xymax[i], l2_xy_max[i])
            intersect_wh=tf.maximum(xy_max-xy_min, 0.)
            intersect_areas=intersect_wh[..., 0]*intersect_wh[..., 1]
            union_areas=preds_areas[i]+l2_area[i]-intersect_areas
            ious=tf.truediv(intersect_areas, union_areas)
            best_ious=tf.reduce_max(ious, axis=3)
            conf_mask_noobj.append(tf.cast(best_ious < seuil_iou_loss, tf.float32)*(1-conf_mask_obj[i]))
        conf_mask_noobj=tf.stack(conf_mask_noobj)

        preds_x=preds_xy[..., 0]
        preds_y=preds_xy[..., 1]
        preds_w=preds_wh[..., 0]
        preds_h=preds_wh[..., 1]
        labels_x=labels_xy[..., 0]
        labels_y=labels_xy[..., 1]
        labels_w=labels_wh[..., 0]
        labels_h=labels_wh[..., 1]

        loss_xy=tf.reduce_sum(conf_mask_obj*(tf.math.square(preds_x-labels_x)+tf.math.square(preds_y-labels_y)), axis=(1, 2, 3))
        loss_wh=tf.reduce_sum(conf_mask_obj*(tf.math.square(preds_w-labels_w)+tf.math.square(preds_h-labels_h)), axis=(1, 2, 3))

        loss_conf_obj=tf.reduce_sum(conf_mask_obj*tf.math.square(preds_conf-conf_mask_obj), axis=(1, 2, 3))
        loss_conf_noobj=tf.reduce_sum(conf_mask_noobj*tf.math.square(preds_conf-conf_mask_obj), axis=(1, 2, 3))

        loss_classe=tf.reduce_sum(tf.math.square(preds_classe-labels_classe), axis=4)
        loss_classe=tf.reduce_sum(conf_mask_obj*loss_classe, axis=(1, 2, 3))
        
        loss = lambda_coord*loss_xy + lambda_coord*loss_wh+loss_conf_obj + lambda_noobj*loss_conf_noobj + loss_classe
        return loss

    return yolo_loss_fixed
