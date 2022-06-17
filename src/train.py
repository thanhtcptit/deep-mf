import os
import time
import shutil
import collections

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from tqdm import tqdm

from src.utils.train_utils import *
from src.models.base import BaseModel
from src.utils import Params, save_json, save_txt, load_json, load_csv, Logger

tf.get_logger().setLevel('INFO')


def create_tf_dataset(data, batch_size, is_train=False):
    n_epochs = None
    if not is_train:
        n_epochs = 1
    tf_dataset = tf.data.experimental.make_csv_dataset(
        data, batch_size=batch_size, num_epochs=n_epochs,
        column_defaults=[tf.int32, tf.int32, tf.float32],
        label_name="label", shuffle=is_train,
        shuffle_buffer_size=20 * batch_size, shuffle_seed=442,
        prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
        num_parallel_reads=tf.data.experimental.AUTOTUNE, sloppy=True)
    return tf_dataset


def evaluate(model, test_dataset, trainer_config):
    metrics = [
        keras.metrics.BinaryAccuracy(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.PrecisionAtRecall(recall=0.8)
    ]
    loss_fn = get_loss_fn(trainer_config["loss_fn"]["type"])(**trainer_config["loss_fn"].get("params", {}))
    mean_loss = keras.metrics.Mean()
    for batch in test_dataset:
        preds = model(batch[0], training=False)
        y_true = tf.expand_dims(batch[1], -1)
        for metric in metrics:
            metric.update_state(y_true, preds)
        loss = loss_fn(y_true, preds)
        mean_loss.update_state(loss)

    results = [mean_loss.result().numpy()]
    for metric in metrics:
        results.append(metric.result().numpy().tolist())
        metric.reset_states()
    return results


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

    log_file = os.path.join(checkpoint_dir, "log.txt")
    logger = Logger(log_file, stdout=False)
    logger.log(f"\n=======================================\n")

    config = Params.from_file(config_path)
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    train_file = os.path.join(data_config["path"]["processed"], "train.csv")
    val_file = os.path.join(data_config["path"]["processed"], "val.csv")
    metadata = load_csv(os.path.join(data_config["path"]["processed"], "metadata.csv"), skip_header=True)

    train_dataset = create_tf_dataset(train_file, trainer_config["batch_size"], is_train=True)
    val_dataset = create_tf_dataset(val_file, trainer_config["batch_size"])

    model_config["num_users"] = int(metadata[0][0]) + 1
    model_config["num_items"] = int(metadata[0][1]) + 1
    model = BaseModel.from_params(model_config).build_graph()
    if recover:
        model.load_weights(weight_dir)
    model.summary()

    optimizer = get_optimizer(trainer_config["optimizer"]["type"])(**trainer_config["optimizer"].get("params", {}))
    loss_fn = get_loss_fn(trainer_config["loss_fn"]["type"])(**trainer_config["loss_fn"].get("params", {}))

    @tf.function
    def train_step(x, y, train_user_emb=True, train_item_emb=True):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss_value = loss_fn(tf.expand_dims(y, -1), preds)

        trainable_weights = []
        if train_user_emb:
            trainable_weights.append(model.trainable_weights[0])
        if train_item_emb:
            trainable_weights.append(model.trainable_weights[1])
        grads = tape.gradient(loss_value, trainable_weights)
        if "grad_clip" in trainer_config:
            grads = [(tf.clip_by_value(grad, clip_value_min=trainer_config["grad_clip"]["min_value"],
                                       clip_value_max=trainer_config["grad_clip"]["max_value"]))
                                       for grad in grads]
        optimizer.apply_gradients(zip(grads, trainable_weights))
        return loss_value

    loss_fn_name = trainer_config["loss_fn"]["type"]
    print(("\n" + " %10s " * 7) % (loss_fn_name, f"val_{loss_fn_name}", "val_acc",
                                 "val_p", "val_r", "val_p@r0.8", "iter"))
    logger.log(("\n" + " %10s " * 7) % ("iter", loss_fn_name, f"val_{loss_fn_name}", "val_acc",
                                        "val_p", "val_r", "val_p@r0.8"))
    results = evaluate(model, val_dataset, trainer_config)
    pbar = tqdm(enumerate(train_dataset))
    pbar.set_description(("%10.4g" * (1 + len(results))) % (0, *results))

    if trainer_config["mode"] == "alt_training":
        flag = 0
        emb_train_step = [trainer_config["num_user_emb_train_step"], trainer_config["num_item_emb_train_step"]]
        swap = emb_train_step[0]
    else:
        flag = -1

    best_val = 0
    for step, batch in pbar:
        if step > trainer_config["num_steps"]:
            pbar.close()
            break
        if flag != -1:
            if step > swap:
                flag = 1 - flag
                swap += emb_train_step[flag]
            loss_value = train_step(batch[0], batch[1], train_user_emb=1 - flag, train_item_emb=flag)
        else:
            loss_value = train_step(batch[0], batch[1])
        if step % trainer_config["display_step"] == 0:
            pbar.set_description(("%10.4g" * (1 + len(results))) % (loss_value, *results))

        if (step + 1) % trainer_config["save_step"] == 0:
            model.save_weights(os.path.join(weight_dir, "latest.ckpt"))

        if (step + 1) % trainer_config["validate_step"] == 0:
            results = evaluate(model, val_dataset, trainer_config)
            pbar.set_description(("%10.4g" * (1 + len(results))) % (loss_value, *results))
            logger.log(("\n" + "%10.4g" * (2 + len(results))) % (step + 1, loss_value, *results))
            if results[-1] > best_val:
                best_val = results[-1]
                model.save_weights(os.path.join(weight_dir, "best.ckpt"))

    results = evaluate(model, val_dataset, trainer_config)
    return results[0]


def eval(checkpoint_dir, dataset_path):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    if not dataset_path:
        dataset_path = os.path.join(data_config["path"]["processed"], "val.csv")
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

        train_file = os.path.join(data_config["path"]["processed"], "train.csv")
        val_file = os.path.join(data_config["path"]["processed"], "val.csv")
        metadata = load_csv(os.path.join(data_config["path"]["processed"], "metadata.csv"), skip_header=True)

        train_dataset = create_tf_dataset(train_file, trainer_config["batch_size"], is_train=True)
        val_dataset = create_tf_dataset(val_file, trainer_config["batch_size"])
        callbacks = []
        if "callbacks" in trainer_config:
            for callback in trainer_config["callbacks"]:
                callbacks.append(get_callback_fn(callback["type"])(**callback["params"]))
        callbacks.append(TFKerasPruningCallback(trial, "val_binary_accuracy"))
        
        model_config["num_users"] = int(metadata[0][0]) + 1
        model_config["num_items"] = int(metadata[0][1]) + 1
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
