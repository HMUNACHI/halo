import math
import tensorflow as tf
from keras import layers

class SinusoidalEmbedding(layers.Layer):
    """
    Sinusoudal Embedding for images
    """
    def __init__(self, 
                 embedding_dims=64,
                 embedding_min_frequency=1.0,
                 embedding_max_frequency=1000.0):
        
        super().__init__()
        self.embedding_min_frequency = embedding_min_frequency
        self.embedding_max_frequency = embedding_max_frequency
        self.embedding_dims = embedding_dims

    def call (self, x):
        start = tf.math.log(self.embedding_min_frequency)
        stop = tf.math.log(self.embedding_max_frequency)
        num = self.embedding_dims // 2
        frequencies = tf.linspace(start, stop, num)
        frequencies = tf.exp(frequencies)
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)
        return embeddings