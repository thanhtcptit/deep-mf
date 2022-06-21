import tensorflow as tf
import tensorflow.keras as keras

from src.models.base import BaseModel


@BaseModel.register("mf")
class MF(BaseModel):
    def __init__(self, num_users, num_items, latent_dim, l2_reg=0, unit_norm_emb=False,
                 use_bias=False, act=None):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.latent_dim = latent_dim
        self.l2_reg = l2_reg
        self.unit_norm_emb = unit_norm_emb
        self.use_bias = use_bias
        self.act = act

        self.user_emb_devices = "gpu" if len(tf.config.list_physical_devices('GPU')) else "cpu"
        if num_users >= 5e5:
            self.user_emb_devices = "cpu"

    def inputs(self):
        return [
            keras.layers.Input(shape=(1), dtype=tf.int32, name="uid"),
            keras.layers.Input(shape=(1), dtype=tf.int32, name="item"),
        ]

    def build_graph(self):
        user_ids, item_ids = self.inputs()

        emb_constrains = None
        if self.unit_norm_emb:
            emb_constrains = keras.constraints.UnitNorm(axis=-1)

        with tf.device(self.user_emb_devices):
            user_embeddings = keras.layers.Embedding(self.num_users, self.latent_dim,
                                                    embeddings_regularizer=keras.regularizers.L2(self.l2_reg),
                                                    embeddings_constraint=emb_constrains, name="user_emb")
            if self.use_bias:
                user_biases = keras.layers.Embedding(self.num_users, 1,
                                                     embeddings_regularizer=keras.regularizers.L2(self.l2_reg),\
                                                     name="user_bias")
        item_embeddings = keras.layers.Embedding(self.num_items, self.latent_dim,
                                                 embeddings_regularizer=keras.regularizers.L2(self.l2_reg),
                                                 embeddings_constraint=emb_constrains, name="item_emb")
        if self.use_bias:
            item_biases = keras.layers.Embedding(self.num_items, 1,
                                                 embeddings_regularizer=keras.regularizers.L2(self.l2_reg),
                                                 name="item_bias")
            global_bias = tf.Variable(0., name="global_bias")

        user_vectors = user_embeddings(user_ids)
        item_vectors = item_embeddings(item_ids)
        scores = tf.math.reduce_sum(user_vectors * item_vectors, axis=-1)
        if self.use_bias:
            scores += user_biases(user_ids) + item_biases(item_ids) + global_bias
        if self.act:
            scores = getattr(tf.math, self.act)(scores)
        return keras.Model(inputs=[user_ids, item_ids], outputs=scores)

    def build_graph_for_hp(self, trial):
        if isinstance(self.latent_dim, list):
            self.latent_dim = trial.suggest_int("latent_dim", self.latent_dim[0], self.latent_dim[1])
        if isinstance(self.l2_reg, list):
            self.l2_reg = trial.suggest_int("l2_reg", self.l2_reg[0], self.l2_reg[1])
        if isinstance(self.unit_norm_emb, list):
            self.unit_norm_emb = trial.suggest_categorical("unit_norm_emb", self.unit_norm_emb)

        return self.build_graph()
