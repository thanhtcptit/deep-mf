import os
import time
import shutil
import collections

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow.keras as keras

from src.utils.train_utils import *
from src.models.base import BaseModel
from src.utils import Params, save_json, save_txt, load_json

tf.get_logger().setLevel('INFO')


def create_tf_dataset(data, batch_size, is_train=False):
    tf_dataset = tf.data.Dataset.from_tensor_slices(data)
    if is_train:
        tf_dataset = tf_dataset.shuffle(buffer_size=batch_size * 100)
    tf_dataset = tf_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return tf_dataset


def train(config_path, checkpoint_dir, recover=False, force=False):
    if not checkpoint_dir:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        checkpoint_dir = os.path.join("train_logs", config_name)
    if os.path.exists(checkpoint_dir):
        if force:
            shutil.rmtree(checkpoint_dir)
        else:
            raise ValueError(f"{checkpoint_dir} already existed")
    weight_dir = os.path.join(checkpoint_dir, "checkpoints")
    os.makedirs(weight_dir, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(checkpoint_dir, "config.json"))

    config = Params.from_file(config_path)
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    train_df = pd.read_csv(data_config["path"]["train"])
    val_df = pd.read_csv(data_config["path"]["val"])

    train_dataset = create_tf_dataset(train_df, trainer_config["batch_size"], is_train=True)
    val_dataset = create_tf_dataset(val_df, trainer_config["batch_size"])

    callbacks = []
    if "callbacks" in trainer_config:
        for callback in trainer_config["callbacks"]:
            if "params" not in callback:
                callback["params"] = {}
            if callback["type"] == "model_checkpoint":
                callback["params"]["filepath"] = os.path.join(weight_dir, "best.ckpt")
            elif callback["type"] == "logging":
                callback["params"]["filename"] = os.path.join(checkpoint_dir, "log.csv")
            callbacks.append(get_callback_fn(callback["type"])(**callback["params"]))

    model = BaseModel.from_params(model_config).build_graph()
    if recover:
        model.load_weights(weight_dir)

    model.compile(
        optimizer=get_optimizer(trainer_config["optimizer"]["type"])(**trainer_config["optimizer"].get("params", {})),
        loss=get_loss_fn(trainer_config["loss_fn"]["type"])(**trainer_config["loss_fn"].get("params", {})),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)])

    model.fit(
        train_dataset, validation_data=val_dataset, epochs=trainer_config["num_epochs"],
        callbacks=callbacks)

    metrics = model.evaluate(val_dataset)
    print(metrics)
    return metrics


def eval(checkpoint_dir, dataset_path):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    if not dataset_path:
        dataset_path = data_config["path"]["val"]
    test_df = pd.read_csv(dataset_path)
    test_dataset = create_tf_dataset(test_df, trainer_config["batch_size"], is_train=True)

    model = BaseModel.from_params(model_config).build_graph()
    model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best.ckpt"))
    model.compile(
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
    metrics = model.evaluate(test_dataset)
    print(metrics)
    return metrics


def hyperparams_search(config_file, num_trials=50, force=False):
    import optuna
    from optuna.integration import TFKerasPruningCallback

    def objective(trial):
        tf.keras.backend.clear_session()

        config = Params.from_file(config_file)
        data_config = config["data"]
        model_config = config["model"]
        trainer_config = config["trainer"]

        train_df = pd.read_csv(data_config["path"]["train"])
        val_df = pd.read_csv(data_config["path"]["val"])

        train_dataset = create_tf_dataset(train_df, trainer_config["batch_size"], is_train=True)
        val_dataset = create_tf_dataset(val_df, trainer_config["batch_size"])

        callbacks = []
        if "callbacks" in trainer_config:
            for callback in trainer_config["callbacks"]:
                callbacks.append(get_callback_fn(callback["type"])(**callback["params"]))
        callbacks.append(TFKerasPruningCallback(trial, "val_binary_accuracy"))

        model = BaseModel.from_params(model_config).build_graph_for_hp(trial)

        optimizer = trial.suggest_categorical("optimizer", trainer_config["optimizer"]["type"])
        lr = trial.suggest_float("lr", trainer_config["optimizer"]["params"]["learning_rate"][0],
                                 trainer_config["optimizer"]["params"]["learning_rate"][1], log=True)
        model.compile(
            optimizer=get_optimizer(optimizer)(learning_rate=lr),
            loss=get_loss_fn(trainer_config["loss_fn"]["type"])(**trainer_config["loss_fn"].get("params", {})),
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                     tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
        model.summary()

        model.fit(
            train_dataset, validation_data=val_dataset, epochs=trainer_config["num_epochs"],
            callbacks=callbacks)

        metrics = model.evaluate(val_dataset)
        return metrics[1]

    study = optuna.create_study(study_name="bert", direction="maximize")
    study.optimize(objective, n_trials=num_trials, gc_after_trial=True,
                   catch=(tf.errors.InvalidArgumentError,))
    print("Number of finished trials: ", len(study.trials))

    df = study.trials_dataframe()
    print(df)

    print("Best trial:")
    trial = study.best_trial

    print(" - Value: ", trial.value)
    print(" - Params: ")
    for key, value in trial.params.items():
        print("  - {}: {}".format(key, value))
