import os
import shutil

import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from tqdm import tqdm
from pprint import pprint
from optuna.integration import TFKerasPruningCallback

from src.utils import *
from src.models.base import BaseModel
from src.models.mf import ReconstructionMF

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
    metrics = []
    if trainer_config["loss_fn"]["type"] in CLASSIFICATION_LOSSES:
        metrics += [
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.PrecisionAtRecall(recall=0.8)
        ]

    loss_fn = build_loss_fn(trainer_config["loss_fn"])
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


def train(config_path, dataset_path, checkpoint_dir, recover=False, force=False):
    config = Params.from_file(config_path)
    config["dataset_path"] = dataset_path
    model_config = config["model"]
    trainer_config = config["trainer"]
    pprint(config.as_dict())

    if not checkpoint_dir:
        dataset_name = get_basename(dataset_path)
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        checkpoint_dir = os.path.join("train_logs", dataset_name, config_name)
    if os.path.exists(checkpoint_dir):
        if force:
            shutil.rmtree(checkpoint_dir)
        else:
            raise ValueError(f"{checkpoint_dir} already existed")
    weight_dir = os.path.join(checkpoint_dir, "checkpoints")
    os.makedirs(weight_dir, exist_ok=True)
    save_json(os.path.join(checkpoint_dir, "config.json"), config.as_dict())
    print("Model checkpoint: ", checkpoint_dir)

    log_file = os.path.join(checkpoint_dir, "log.txt")
    logger = Logger(log_file, stdout=False)
    logger.log(f"\n=======================================\n")

    train_file = os.path.join(dataset_path, "train.csv")
    val_file = os.path.join(dataset_path, "val.csv")
    metadata = load_csv(os.path.join(dataset_path, "metadata.csv"), skip_header=True)

    train_dataset = create_tf_dataset(train_file, trainer_config["batch_size"], is_train=True)
    val_dataset = create_tf_dataset(val_file, trainer_config["batch_size"])

    model_config["num_users"] = int(metadata[0][0]) + 1
    model_config["num_items"] = int(metadata[0][1]) + 1
    model = BaseModel.from_params(model_config).build_graph()
    if recover:
        model.load_weights(weight_dir)
    model.summary()

    loss_fn = build_loss_fn(trainer_config["loss_fn"])
    optimizer = build_optimizer(trainer_config["optimizer"])
    grad_clip_fn = None
    if "grad_clip" in trainer_config:
        grad_clip_fn = build_gradient_clipping_fn(trainer_config["grad_clip"])

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
        if grad_clip_fn:
            grads = grad_clip_fn(grads)
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


def test(checkpoint_dir, test_dataset_path):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    dataset_path = config["dataset_path"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    if not test_dataset_path:
        test_dataset_path = os.path.join(dataset_path, "val.csv")
    test_df = pd.read_csv(test_dataset_path)
    test_dataset = create_tf_dataset(test_df, trainer_config["batch_size"], is_train=True)

    metadata = load_csv(os.path.join(dataset_path, "metadata.csv"), skip_header=True)
    model_config["num_users"] = int(metadata[0][0]) + 1
    model_config["num_items"] = int(metadata[0][1]) + 1
    model = BaseModel.from_params(model_config).build_graph()
    model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best.ckpt"))

    metrics = []
    if trainer_config["loss_fn"]["type"] in CLASSIFICATION_LOSSES:
        metrics += [
            keras.metrics.BinaryAccuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.PrecisionAtRecall(recall=0.8)
        ]
    model.compile(metrics=metrics)
    metrics = model.evaluate(test_dataset)
    print(metrics)
    return metrics


def test_keyword(checkpoint_dir, test_dataset_path, additional_dataset_path=None, recontruction_config=None):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    dataset_path = config["dataset_path"]
    model_config = config["model"]

    metadata = load_csv(os.path.join(dataset_path, "metadata.csv"), skip_header=True)
    model_config["num_users"] = int(metadata[0][0]) + 1
    model_config["num_items"] = int(metadata[0][1]) + 1
    model = BaseModel.from_params(model_config).build_graph()
    model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best.ckpt"))

    user_emb = model.layers[2].weights[0].numpy()
    item_emb = model.layers[3].weights[0].numpy()

    if not test_dataset_path:
        test_dataset_path = os.path.join(dataset_path, "val.csv")

    kw_to_item_list = load_json(os.path.join(dataset_path, "kw_map.json"))
    user_map_file = os.path.join(dataset_path, "user_map.csv")
    user_data = pd.read_csv(user_map_file, dtype={"uid": str, "index": int}, na_values=0)
    uid_to_index = {k: v for k, v in zip(user_data["uid"].tolist(), user_data["index"].tolist())}

    item_map_file = os.path.join(dataset_path, "item_map.csv")
    item_to_index = {k: int(v) for k, v in load_dict(
        item_map_file, sep=",", skip_header=True).items()}

    new_users_vector = {}
    if additional_dataset_path and recontruction_config is not None:
        print("Calculating latent vectors for new users")
        if isinstance(recontruction_config, str):
            recontruction_config = load_json(recontruction_config)
        pprint(recontruction_config)

        recon_data_config = recontruction_config["data"]
        recon_model_config = recontruction_config["model"]
        recon_trainer_config = recontruction_config["trainer"]

        loss_fn = build_loss_fn(recon_trainer_config["loss_fn"])
        optimizer = build_optimizer(recon_trainer_config["optimizer"])
        if "grad_clip" in recon_trainer_config:
            grad_clip_fn = build_gradient_clipping_fn(recon_trainer_config["grad_clip"])
        else:
            grad_clip_fn = None

        mf = ReconstructionMF(item_emb, recon_model_config["act"], optimizer, loss_fn, grad_clip_fn,
                              l2_reg=recon_model_config["l2_reg"], reconstruct_iter=recon_trainer_config["reconstruct_iter"])
        addtional_data = pd.read_csv(additional_dataset_path, dtype={"uid": str, "item": str, "label": np.float16})
        addtional_data = addtional_data.groupby("uid").agg(lambda x: list(x)).reset_index()

        item_set = set(list(item_to_index.values()))
        for i, r in tqdm(addtional_data.iterrows()):
            if r["uid"] in uid_to_index:
                continue
            user_data = [(item_to_index[i], l) for i, l in zip(r["item"], r["label"]) if i in item_to_index]
            pos_inds = [x[0] for x in user_data]
            labels = [x[1] for x in user_data]
            if len(labels) < 1:
                continue

            num_neg_samples = int(np.ceil(recon_data_config["neg_to_pos_ratio"] * len(labels)))
            neg_candidates = list(item_set - set(pos_inds))
            if num_neg_samples > len(neg_candidates):
                num_neg_samples = len(neg_candidates)
            neg_inds = np.random.choice(neg_candidates, num_neg_samples, replace=False).tolist()

            train_ids = pos_inds + neg_inds
            train_labels = labels + [0] * len(neg_inds)
            user_tf_dataset = tf.data.Dataset.from_tensor_slices((train_ids, train_labels)).batch(recon_trainer_config["batch_size"])
            user_vector, _ = mf.reconstruct(user_tf_dataset)
            new_users_vector[r["uid"]] = user_vector
        print(len(new_users_vector))

    top_k = [3, 5]
    ctr, new_users_ctr = [0, 0], [0, 0]
    total_row, new_users_total_row = 0, 0
    count_na = 0
    user_list = []
    with open(test_dataset_path) as f:
        f.readline()
        for line in tqdm(f):
            data = line.strip().split("\t")
            uid, kw, item = data[0], data[3], data[4]
            if (uid not in uid_to_index and uid not in new_users_vector) or \
                item not in item_to_index or kw not in kw_to_item_list:  
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
            if uid in uid_to_index:
                new_user = False
                user_vector = np.expand_dims(user_emb[uid_to_index[uid]], axis=0)
            else:
                new_user = True
                user_vector = new_users_vector[uid]
                new_users_total_row += 1

            scores = user_vector @ candidate_item_vector.T
            sorted_scores = np.argsort(scores[0])[::-1]
            for i, k in enumerate(top_k):
                top_k_pos = sorted_scores[:k].tolist()
                if clicked_ind_pos in top_k_pos:
                    ctr[i] += 1
                    if new_user:
                        new_users_ctr[i] += 1
            total_row += 1

    print("Total: ", total_row)
    print("NA: ", count_na)
    for i, k in enumerate(top_k):
        print(f"[Total] CTR@{k}: {ctr[i] / total_row:.04f}")
        if len(new_users_vector):
            print(f"[New users] CTR@{k}: {new_users_ctr[i] / new_users_total_row:.04f}")
    return ctr[0] / total_row


def hyperparams_search_training(config_file, dataset_path, test_dataset_path, additional_dataset_path=None,
                                num_trials=100, force=False):
    def training(trial):
        tf.keras.backend.clear_session()

        dataset_name = get_basename(dataset_path)
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
        checkpoint_dir = f"/tmp/{dataset_name}/{config_name}"
        trial_config_file = os.path.join(f"/tmp/hyp_{get_current_time_str()}.json")
        save_json(trial_config_file, config)

        best_val = train(trial_config_file, dataset_path, checkpoint_dir, force=force)
        if test_dataset_path:
            best_val = test_keyword(checkpoint_dir, test_dataset_path, additional_dataset_path)
        return best_val

    study = optuna.create_study(study_name="mf_training", direction="maximize")
    study.optimize(training, n_trials=num_trials, gc_after_trial=True,
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


def hyperparams_search_reconstruction(config_file, checkpoint_dir, test_dataset_path, additional_dataset_path,
                                      num_trials=100):
    def reconstruction(trial):
        tf.keras.backend.clear_session()

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

        config.pop("hyp")
        best_val = test_keyword(checkpoint_dir, test_dataset_path, additional_dataset_path, config)
        return best_val

    study = optuna.create_study(study_name="mf_reconstruction", direction="maximize")
    study.optimize(reconstruction, n_trials=num_trials, gc_after_trial=True,
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
