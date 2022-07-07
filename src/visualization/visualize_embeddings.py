import os

import numpy as np
from annoy import AnnoyIndex

from src.utils import *
from src.models.base import BaseModel


def get_similar_items(checkpoint_dir, force=False):
    config = Params.from_file(os.path.join(checkpoint_dir, "config.json"))
    dataset_path = config["dataset_path"]
    model_config = config["model"]

    item_map_file = os.path.join(dataset_path, "item_map.csv")
    item_to_index = {k: int(v) for k, v in load_dict(item_map_file, sep=",", skip_header=True).items()}
    index_to_item = {v: k for k, v in item_to_index.items()}

    ann = AnnoyIndex(model_config["latent_dim"], "euclidean")
    ann_save_path = os.path.join(checkpoint_dir, "item_tree.ann")
    if not os.path.exists(ann_save_path) or (os.path.exists(ann_save_path) and force):
        metadata = load_csv(os.path.join(dataset_path, "metadata.csv"), skip_header=True)
        model_config["num_users"] = int(metadata[0][0]) + 1
        model_config["num_items"] = int(metadata[0][1]) + 1
        model = BaseModel.from_params(model_config).build_graph()
        model.load_weights(os.path.join(checkpoint_dir, "checkpoints/best.ckpt"))

        item_emb = model.layers[3].weights[0].numpy()
        for i in range(item_emb.shape[0]):
            ann.add_item(i, item_emb[i])

        ann.build(10)
        ann.save(ann_save_path)
    else:
        ann.load(ann_save_path)

    n_items = 10
    for i in range(n_items):
        ind = np.random.randint(1, len(index_to_item) + 1)
        closest_inds = ann.get_nns_by_item(ind, n=10)
        print(f"s{i} = [" + ", ".join([f'"{index_to_item[si]}"' for si in closest_inds]) + "]")
