import ssl
import tensorflow as tf
from keras import layers
from tensorflow import keras

ssl._create_default_https_context = ssl._create_unverified_context

class KID(keras.metrics.Metric):
    def __init__(self, name, image_size, kid_image_size=75, **kwargs):
        super().__init__(name=name, **kwargs)
        """
        Kernel Inception Distance for estimating the similarities 
        of the distribution of 2 samples.
        """
        self.kid_image_size = kid_image_size
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")
        self.encoder = keras.Sequential([keras.Input(shape=(image_size, image_size, 3)),
                                         layers.Rescaling(255.0),
                                         layers.Resizing(height=kid_image_size, width=kid_image_size),
                                         layers.Lambda(keras.applications.inception_v3.preprocess_input),
                                         keras.applications.InceptionV3(include_top=False,
                                                                        input_shape=(kid_image_size, kid_image_size, 3),
                                                                        weights="imagenet",),
                                         layers.GlobalAveragePooling2D()],
                                         name="inception_encoder")

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(generated_features, generated_features)
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)

        numerator = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size)))
        denominator = (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_real = numerator / denominator

        numerator_2 = tf.reduce_sum(kernel_generated * (1.0 - tf.eye(batch_size)))
        denominator_2 = (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_generated = numerator_2 / denominator_2

        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
        
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()