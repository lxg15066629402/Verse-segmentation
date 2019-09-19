from __future__ import print_function
from keras import backend as K
import numpy as np

# def weight_log_loss(y_true, y_pred):
#     y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#     y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
#
#     weight = np.array([0.01, 10.0, 10.0, 10.0, 10.0,
#                        10.0, 10.0, 10.0, 10.0, 10.0,
#                        10.0, 10.0, 10.0, 10.0, 10.0,
#                        10.0, 10.0, 10.0, 10.0, 10.0,
#                        10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
#     weight = K.variable(weight)
#     loss = y_true * K.log(y_pred) * weight
#     loss = K.mean(-K.sum(loss, -1))
#     return loss


def weight_log_loss(y_true, y_pred):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    weight = np.array([0.01, 10.0, 10.0, 10.0, 10.0,
                       10.0, 10.0, 10.0, 10.0, 10.0,
                       10.0, 10.0, 10.0, 10.0, 10.0,
                       10.0, 10.0, 10.0, 10.0, 10.0,
                       10.0, 10.0, 10.0, 10.0, 10.0, 30.0])
    weight = K.variable(weight)
    loss = y_true * K.log(y_pred) * weight
    loss = K.mean(-K.sum(loss, -1))
    return loss


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
#
# def dice_coefficient_loss(y_true, y_pred):
#     return -dice_coefficient(y_true, y_pred)


# def dice_loss(y_true, y_pred):
#     '''
#     computes the sum of two losses : generalised dice loss and weighted cross entropy
#     '''
#     y_true_f = K.reshape(y_true, shape=(-1, 2))
#     y_pred_f = K.reshape(y_pred, shape=(-1, 2))
#     sum_p = K.sum(y_pred_f, axis=-2)
#     sum_r = K.sum(y_true_f, axis=-2)
#     sum_pr = K.sum(y_true_f * y_pred_f, axis=-2)
#     weights = K.pow(K.square(sum_r) + K.epsilon(), -1)  # cross entropy loss
#     generalised_dice_numerator = 2 * K.sum(weights * sum_pr)
#     generalised_dice_denominator = K.sum(weights * (sum_r + sum_p))
#     generalised_dice_score = (generalised_dice_numerator + K.epsilon()) / (generalised_dice_denominator + K.epsilon())
#     GDL = 1 - generalised_dice_score
#
#     # return GDL + weight_log_loss(y_true, y_pred)
#     return GDL
#

def dice_loss(y_true, y_pred):
    '''
    computes the sum of two losses : generalised dice loss and weighted cross entropy
    '''
    y_true_f = K.reshape(y_true, shape=(-1, 26))
    y_pred_f = K.reshape(y_pred, shape=(-1, 26))
    sum_p = K.sum(y_pred_f, axis=-2)
    sum_r = K.sum(y_true_f, axis=-2)
    sum_pr = K.sum(y_true_f * y_pred_f, axis=-2)
    weights = K.pow(K.square(sum_r) + K.epsilon(), -1)  # cross entropy loss
    generalised_dice_numerator = 2 * K.sum(weights * sum_pr)
    generalised_dice_denominator = K.sum(weights * (sum_r + sum_p))
    generalised_dice_score = (generalised_dice_numerator + K.epsilon()) / (generalised_dice_denominator + K.epsilon())
    GDL = 1 - generalised_dice_score

    return GDL + weight_log_loss(y_true, y_pred)
    # return GDL


def dice(y_true, y_pred):
    # computes the dice score on two tensors

    sum_p = K.sum(y_pred, axis=0)
    sum_r = K.sum(y_true, axis=0)
    sum_pr = K.sum(y_true * y_pred, axis=0)
    dice_numerator = 2 * sum_pr
    dice_denominator = sum_r + sum_p
    dice_score = (dice_numerator + K.epsilon()) / (dice_denominator + K.epsilon())

    return dice_score


def dice_score_metric(y_true, y_pred):
    '''
    computes the sum of two losses : generalised dice loss and weighted cross entropy
    '''

    y_true_f = K.reshape(y_true, shape=(-1, 26))
    y_pred_f = K.reshape(y_pred, shape=(-1, 26))

    y_whole = K.sum(y_true_f[:, 1:], axis=1)
    p_whole = K.sum(y_pred_f[:, 1:], axis=1)

    dice_score_m = dice(y_whole, p_whole)

    return dice_score_m


# def dice_score_metric(y_true, y_pred):
#     '''
#     computes the sum of two losses : generalised dice loss and weighted cross entropy
#     '''
#
#     y_true_f = K.reshape(y_true, shape=(-1, 2))
#     y_pred_f = K.reshape(y_pred, shape=(-1, 2))
#
#     y_whole = K.sum(y_true_f[:, 1:], axis=1)
#     p_whole = K.sum(y_pred_f[:, 1:], axis=1)
#
#     dice_score_m = dice(y_whole, p_whole)
#
#     return dice_score_m
