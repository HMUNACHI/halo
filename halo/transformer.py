from keras import layers
from tensorflow import keras

class VisionTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        """
        Vision Transformer encoder block
        """
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = layers.MultiHeadAttention(num_heads, head_size)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = layers.MultiHeadAttention(num_heads, head_size)
        self.norm3 = layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        normalized_inputs = self.norm1(inputs)
        x = self.attn1(normalized_inputs, normalized_inputs) + inputs
        return self.dense(self.geglu(self.norm3(x))) + x


class GEGLU(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = keras.layers.Dense(output_dim * 2)

    def call(self, inputs):
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = keras.activations.tanh(gate * 0.7978845608 * (1 + 0.044715 * (gate**2)))
        return x * 0.5 * gate * (1 + tanh_res)