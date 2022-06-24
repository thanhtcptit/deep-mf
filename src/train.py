import os
import glob
import time
import shutil
import collections

import numpy as np
import pandas as pd
import tensorflow as tf
from src.utils.file_utils import save_json
import tensorflow.keras as keras

from tqdm import tqdm
from pprint import pprint

from src.utils.train_utils import *
from src.models.base import BaseModel
from src.utils import Params, get_current_time_str, load_json, load_dict, load_csv, Logger

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
    if trainer_config["loss_fn"]["type"] in ["bce", "wbce", "cce", "focal"]:
        metrics = [
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.PrecisionAtRecall(recall=0.8)
        ]
    else:
        metrics = []
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
    config = Params.from_file(config_path)
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]
    pprint(config.as_dict())

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
    shutil.copy(config_path, os.path.join(checkpoint_dir, "config.json"))
    print("Model checkpoint: ", checkpoint_dir)

    log_file = os.path.join(checkpoint_dir, "log.txt")
    logger = Logger(log_file, stdout=False)
    logger.log(f"\n=======================================\n")

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
        alt_training_flag = 0
        emb_train_step = [trainer_config["num_user_emb_train_step"], trainer_config["num_item_emb_train_step"]]
        swap_training = emb_train_step[0]
    else:
        alt_training_flag = -1

    best_loss = float("inf")
    for step, batch in pbar:
        if step > trainer_config["num_steps"]:
            pbar.close()
            break
        if alt_training_flag != -1:
            if step > swap_training:
                alt_training_flag = 1 - alt_training_flag
                swap_training += emb_train_step[alt_training_flag]
            loss_value = train_step(batch[0], batch[1], train_user_emb=1 - alt_training_flag,
                                    train_item_emb=alt_training_flag)
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
            if best_loss > results[0]:
                best_loss = results[0]
                model.save_weights(os.path.join(weight_dir, "best.ckpt"))

    return -best_loss


def test(checkpoint_dir, dataset_path):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    if not dataset_path:
        dataset_path = os.path.join(data_config["path"]["processed"], "val.csv")
    test_df = pd.read_csv(dataset_path)
    test_dataset = create_tf_dataset(test_df, trainer_config["batch_size"], is_train=True)

    metadata = load_csv(os.path.join(data_config["path"]["processed"], "metadata.csv"), skip_header=True)
    model_config["num_users"] = int(metadata[0][0]) + 1
    model_config["num_items"] = int(metadata[0][1]) + 1
    model = BaseModel.from_params(model_config).build_graph()
    model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best.ckpt"))
    model.compile(
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall(), tf.keras.metrics.PrecisionAtRecall(recall=0.8)])
    metrics = model.evaluate(test_dataset)
    print(metrics)
    return metrics


def test_keyword(checkpoint_dir, dataset_path):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    data_config = config["data"]
    model_config = config["model"]

    metadata = load_csv(os.path.join(data_config["path"]["processed"], "metadata.csv"), skip_header=True)
    model_config["num_users"] = int(metadata[0][0]) + 1
    model_config["num_items"] = int(metadata[0][1]) + 1
    model = BaseModel.from_params(model_config).build_graph()
    model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best.ckpt"))

    user_emb = model.layers[2].weights[0].numpy()
    item_emb = model.layers[3].weights[0].numpy()

    if not dataset_path:
        dataset_path = os.path.join(data_config["path"]["processed"], "val.csv")

    kw_to_item_list = load_json(os.path.join(data_config["path"]["processed"], "kw_map.json"))
    user_map_file = os.path.join(data_config["path"]["processed"], "user_map.csv")
    user_data = pd.read_csv(user_map_file, dtype={"uid": str, "index": int}, na_values=0)
    uid_to_index = {k: v for k, v in zip(user_data["uid"].tolist(), user_data["index"].tolist())}

    item_map_file = os.path.join(data_config["path"]["processed"], "item_map.csv")
    item_to_index = {k: int(v) for k, v in load_dict(
        item_map_file, sep=",", skip_header=True).items()}

    top_k = [3, 5]
    ctr = [0, 0]
    total_row = 0
    count_na = 0
    user_list = []
    with open(dataset_path) as f:
        f.readline()
        for line in tqdm(f):
            data = line.strip().split("\t")
            uid, kw, item = data[0], data[3], data[4]
            if uid not in uid_to_index or item not in item_to_index or kw not in kw_to_item_list:  
                count_na += 1
                continue
            user_list.append(uid)

            kw_item_list = kw_to_item_list[kw]
            candidate_item_inds = [item_to_index[str(i)] for i in kw_item_list if str(i) in item_to_index]
            clicked_ind = item_to_index[item]
            if clicked_ind not in candidate_item_inds:
                continue

            clicked_ind_pos = candidate_item_inds.index(clicked_ind)
            candidate_item_vector = item_emb[candidate_item_inds]
            user_vector = np.expand_dims(user_emb[uid_to_index[uid]], axis=0)
            scores = user_vector @ candidate_item_vector.T
            sorted_scores = np.argsort(scores[0])[::-1]
            for i, k in enumerate(top_k):
                top_k_pos = sorted_scores[:k].tolist()
                if clicked_ind_pos in top_k_pos:
                    ctr[i] += 1
            total_row += 1

    print("Total: ", total_row)
    print("NA: ", count_na)
    for i, k in enumerate(top_k):
        print(f"CTR@{k}: {ctr[i] / total_row:.04f}")
    return ctr[0] / total_row


def hyperparams_search(config_file, dataset_path, num_trials=50, force=False):
    import optuna
    from optuna.integration import TFKerasPruningCallback

    def objective(trial):
        tf.keras.backend.clear_session()
        
        config_name = os.path.splitext(os.path.basename(config_file))[0]
        config = load_json(config_file)
        hyp_config = config["hyp"]
        for k, v in hyp_config.items():
            k_list = k.split(".")
            d = config
            for i in range(len(k_list) - 1):
                d = d[k_list[i]]
            if v["type"] == "int":
                val = trial.suggest_int(k, v["range"][0], v["range"][1])
            elif v["type"] == "float":
                val = trial.suggest_float(k, v["range"][0], v["range"][1], log=v.get("log", False))
            elif v["type"] == "categorical":
                val = trial.suggest_categorical(k, v["values"])
            d[k_list[-1]] = val
            config_name += f"_{k_list[-1]}-{val}"

        config.pop("hyp")
        checkpoint_dir = f"/tmp/{config_name}"
        trial_config_file = os.path.join(f"/tmp/hyp_{get_current_time_str()}.json")
        save_json(trial_config_file, config)

        best_val = train(trial_config_file, checkpoint_dir, force=force)
        if dataset_path:
            best_val = test_keyword(checkpoint_dir, dataset_path)
        return best_val

    study = optuna.create_study(study_name="mf", direction="maximize")
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
