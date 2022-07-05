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

        with tf.device(self.user_emb_devices):
            user_embeddings = keras.layers.Embedding(self.num_users, self.latent_dim,
                                                    embeddings_regularizer=keras.regularizers.L2(self.l2_reg),
                                                    name="user_emb")

            if self.use_bias:
                user_biases = keras.layers.Embedding(self.num_users, 1,
                                                     embeddings_regularizer=keras.regularizers.L2(self.l2_reg),\
                                                     name="user_bias")
        item_embeddings = keras.layers.Embedding(self.num_items, self.latent_dim,
                                                 embeddings_regularizer=keras.regularizers.L2(self.l2_reg),
                                                 name="item_emb")
        if self.use_bias:
            item_biases = keras.layers.Embedding(self.num_items, 1,
                                                 embeddings_regularizer=keras.regularizers.L2(self.l2_reg),
                                                 name="item_bias")
            global_bias = tf.Variable(0., name="global_bias")

        user_vectors = user_embeddings(user_ids)
        item_vectors = item_embeddings(item_ids)
        if self.unit_norm_emb:
            user_vectors = keras.constraints.UnitNorm(axis=-1)(user_vectors)
            item_vectors = keras.constraints.UnitNorm(axis=-1)(item_vectors)
        scores = tf.math.reduce_sum(user_vectors * item_vectors, axis=-1)
        if self.use_bias:
            scores += tf.squeeze(user_biases(user_ids) + item_biases(item_ids), axis=-1) + global_bias
        if self.act:
            scores = getattr(tf.math, self.act)(scores)
        return keras.Model(inputs=[user_ids, item_ids], outputs=scores)


class UserEmbedding(keras.layers.Layer):
    def __init__(self, latent_dim, regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.regularizer = regularizer

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(1, self.latent_dim), regularizer=self.regularizer,
            dtype=tf.float32, name='user_emb_weight')
        super().build(input_shape)

    def call(self, _):
        return self.embedding

    def compute_output_shape(self):
        return (1, self.latent_dim)


class ReconstructionMF:
    def __init__(self, item_matrix, act, optimizer, loss_fn, grad_clip=None, l2_reg=0, reconstruct_iter=1):
        self.grad_clip = grad_clip
        self.reconstruct_iter = reconstruct_iter

        self.num_items = item_matrix.shape[0]
        self.latent_dim = item_matrix.shape[1]

        item_input = keras.layers.Input(shape=(1), dtype=tf.int32, name='item')

        user_embedding_layer = UserEmbedding(
            self.latent_dim, regularizer=keras.regularizers.l2(l2_reg), name='user_emb')
        self.user_embedding_init_weights = keras.initializers.GlorotUniform(seed=442)(shape=(1, self.latent_dim))

        item_embedding_layer = keras.layers.Embedding(
            self.num_items, self.latent_dim, name='item_emb',
                embeddings_initializer=keras.initializers.Constant(item_matrix))

        user_vector = user_embedding_layer(item_input)
        item_vector = keras.layers.Flatten(name='flatten')(item_embedding_layer(item_input))
        pred = tf.math.reduce_sum(user_vector * item_vector, axis=-1)
        if act:
            pred = getattr(tf.math, act)(pred)
        self.model = keras.Model(inputs=item_input, outputs=pred)

        self.loss_fn   = loss_fn
        self.optimizer = optimizer

    def reconstruct(self, data):
        self.model.layers[1].trainable = False
        self.model.layers[2].set_weights([self.user_embedding_init_weights])

        for _ in range(self.reconstruct_iter):
            for batch in data:
                with tf.GradientTape() as tape:
                    pred = self.model(batch[0])
                    loss = self.loss_fn(batch[1], pred)
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    if self.grad_clip:
                        grads = self.grad_clip(grads)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return self.model.layers[2].weights[0].numpy(), loss.numpy()
