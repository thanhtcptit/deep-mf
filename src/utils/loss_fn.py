import tensorflow as tf

E = 2.71828


def log(x, base=E):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


def weighted_mse(multiply=1):
    def func(y_true, y_pred):
        pseudo_label = tf.where(y_true <= 0, tf.zeros_like(y_true), tf.ones_like(y_true))
        loss = tf.math.square(pseudo_label - y_pred) * (multiply * log(y_true + 1) + 1)
        return tf.math.reduce_mean(loss)
    return func


def weighted_bce(multiply=1):
    def func(y_true, y_pred):
        pseudo_label = tf.where(y_true <= 0, tf.zeros_like(y_true), tf.ones_like(y_true))
        loss = tf.keras.backend.binary_crossentropy(pseudo_label, y_pred) * (multiply * log(y_true + 1) + 1)
        return tf.math.reduce_mean(loss)
    return func


def pairwise_loss(N, margin):
    def func(y_true, y_pred):
        d = tf.linalg.norm(y_pred[:, :N] - y_pred[:, N:], axis=-1)
        loss = y_true * d + (1 - y_true) * tf.math.maximum(margin - d, 0)
        return tf.math.reduce_sum(loss, axis=None)
    return func


def triplet_loss(N, margin):
    def func(y_true, y_pred):
        anchor = y_pred[:, :N]
        positive = y_pred[:, N: 2 * N]
        negative = y_pred[:, 2 * N:]

        pos_dist = tf.math.reduce_sum(tf.math.square(anchor - positive), axis=1)
        neg_dist = tf.math.reduce_sum(tf.math.square(anchor - negative), axis=1)
        loss = tf.math.maximum(pos_dist - neg_dist + margin, 0.)    
        return tf.math.reduce_sum(loss, axis=None)
    return func


def lossless_triplet_loss(N, beta, epsilon=1e-8):
    def func(y_true, y_pred):
        anchor   = y_pred[:, :N]
        positive = y_pred[:, N: 2 * N]
        negative = y_pred[:, 2 * N:]

        pos_dist = tf.math.reduce_sum(tf.math.square(anchor - positive), axis=1)
        neg_dist = tf.math.reduce_sum(tf.math.square(anchor - negative), axis=1)
        pos_dist = -tf.math.log(-(pos_dist / beta) + 1 + epsilon)
        neg_dist = -tf.math.log(-((N - neg_dist) / beta) + 1 + epsilon)
        loss = neg_dist + pos_dist
        return tf.math.reduce_sum(loss, axis=None)
    return func


def focal_loss(gamma=2.0, alpha=0.2):
    def focal_loss_fn(y_true, y_pred):
        pt_1 = tf.where(y_true >= 1, y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(y_true <= 0, y_pred, tf.zeros_like(y_pred))
        return -tf.math.reduce_mean(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
            - tf.math.reduce_mean((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0))
    return focal_loss_fn


def weighted_focal_loss(gamma=2.0, alpha=0.2, multiply=1):
    def focal_loss_fn(y_true, y_pred):
        pt_1 = tf.where(y_true >= 1, y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(y_true <= 0, y_pred, tf.zeros_like(y_pred))
        w_ = log(tf.where(y_true >= 1, y_true, tf.ones_like(y_true))) + 1
        return -tf.math.reduce_mean(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1) * (multiply * w_ + 1)) \
            - tf.math.reduce_mean((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0) * (multiply * w_ + 1))
    return focal_loss_fn


CLASSIFICATION_LOSSES = ["bce", "wbce", "cce", "focal", "wfocal"]
REGRESSION_LOSS = ["mse", "wmse"]
METRIC_LOSS = ["pairwise", "triplet", "lossless_triplet"]

def build_loss_fn(config):
    import copy
    
    config_copy = copy.deepcopy(config)

    loss_fn_dict = {
        "bce": tf.keras.losses.BinaryCrossentropy,
        "wbce": weighted_bce,
        "cce": tf.keras.losses.CategoricalCrossentropy,
        "mse": tf.keras.losses.MeanSquaredError,
        "wmse": weighted_mse,
        "focal": focal_loss,
        "wfocal": weighted_focal_loss,
        "pairwise": pairwise_loss,
        "triplet": triplet_loss,
        "lossless_triplet": lossless_triplet_loss
    }
    return loss_fn_dict[config_copy.pop("type")](**config_copy)
