import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import backend as K

E = 2.71828


def bell_curve_lr_scheduler(lr_start=1e-5, lr_max=5e-5, lr_min=1e-6, lr_rampup_epochs=5, 
                            lr_sustain_epochs=0, lr_exp_decay=.87):
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * \
                lr_exp_decay ** (epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


def decay_lr_scheduler(lr=1e-5, decay_rate=0.8):
    def lrfn(epoch):
        return lr / (decay_rate * epoch + 1)
    return lrfn


def get_lr_scheduler_callback(name, params):
    lr_scheduler_dict = {
        "decay": decay_lr_scheduler,
        "bell": bell_curve_lr_scheduler
    }
    return tf.keras.callbacks.LearningRateScheduler(lr_scheduler_dict[name](**params))


def get_optimizer(name):
    optimizer_dict = {
        "adam": keras.optimizers.Adam,
        "sgd": keras.optimizers.SGD,
        "rms": keras.optimizers.RMSprop
    }
    return optimizer_dict[name]


def log(x, base=E):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


def weighted_mse():
    def func(y_true, y_pred):
        pseudo_label = tf.where(y_true == 0, y_true, tf.ones_like(y_true))
        d = tf.square(pseudo_label - y_pred) * (2 * log(y_true + 1, 10) + 1)
        return tf.reduce_mean(d)
    return func


def weighted_bce():
    def func(y_true, y_pred):
        pseudo_label = tf.where(y_true == 0, y_true, tf.ones_like(y_true))
        d = tf.keras.backend.binary_crossentropy(pseudo_label, y_pred) * (2 * log(y_true + 1, 10) + 1)
        return tf.reduce_mean(d)
    return func


def focal_loss(gamma=2.0, alpha=0.2):
    def focal_loss_fn(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) \
            * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fn


def get_loss_fn(name):
    loss_fn_dict = {
        "bce": keras.losses.BinaryCrossentropy,
        "wbce": weighted_bce,
        "cce": keras.losses.CategoricalCrossentropy,
        "mse": keras.losses.MeanSquaredError,
        "wmse": weighted_mse,
        "focal": focal_loss
    }
    return loss_fn_dict[name]


def get_callback_fn(name):
    callback_fn_dict = {
        "early_stopping": keras.callbacks.EarlyStopping,
        "model_checkpoint": keras.callbacks.ModelCheckpoint,
        "logging": keras.callbacks.CSVLogger,
        "lr_scheduler": get_lr_scheduler_callback
    }
    return callback_fn_dict[name]