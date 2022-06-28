import tensorflow as tf


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
        "sgd": tf.keras.optimizers.SGD,
        "adam": tf.keras.optimizers.Adam,
        "rms": tf.keras.optimizers.RMSprop
    }
    return optimizer_dict[name]


def get_callback_fn(name):
    callback_fn_dict = {
        "early_stopping": tf.keras.callbacks.EarlyStopping,
        "model_checkpoint": tf.keras.callbacks.ModelCheckpoint,
        "logging": tf.keras.callbacks.CSVLogger,
        "lr_scheduler": get_lr_scheduler_callback
    }
    return callback_fn_dict[name]