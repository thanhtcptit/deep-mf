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
        if unit_norm_emb:
            self.emb_constrains = keras.constraints.UnitNorm(axis=-1)
        else:
            self.emb_constrains = None
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

        with tf.device(self.user_emb_devices):
            user_embeddings = keras.layers.Embedding(self.num_users, self.latent_dim,
                                                    embeddings_regularizer=keras.regularizers.L2(self.l2_reg),
                                                    embeddings_constraint=self.emb_constrains)
            if self.use_bias:
                user_biases = keras.layers.Embedding(self.num_users, 1,
                                                     embeddings_regularizer=keras.regularizers.L2(self.l2_reg))
        item_embeddings = keras.layers.Embedding(self.num_items, self.latent_dim,
                                                 embeddings_regularizer=keras.regularizers.L2(self.l2_reg),
                                                 embeddings_constraint=self.emb_constrains)
        if self.use_bias:
            item_biases = keras.layers.Embedding(self.num_items, 1,
                                                 embeddings_regularizer=keras.regularizers.L2(self.l2_reg))
            global_bias = tf.Variable(0.)

        user_vectors = user_embeddings(user_ids)
        item_vectors = item_embeddings(item_ids)
        scores = tf.math.reduce_sum(user_vectors * item_vectors, axis=-1)
        if self.use_bias:
            scores += user_biases(user_ids) + item_biases(item_ids) + global_bias
        if self.act:
            scores = getattr(tf.math, self.act)(scores)
        return keras.Model(inputs=[user_ids, item_ids], outputs=scores)
