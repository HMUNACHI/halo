import tensorflow as tf
import matplotlib.pyplot as plt

from keras import layers
from tensorflow import keras
from metrics import KID
from generator import TransformerUNet


class Halo(keras.Model):
    def __init__(self, 
                 image_size, 
                 widths, 
                 block_depth,
                 min_signal_rate=0.02,
                 max_signal_rate=0.95,
                 kid_diffusion_steps=5,
                 kid_image_size=75,
                 ema=0.999):
        
        super().__init__()
        """
        Halo Image Generating Diffusion Model
        """
        self.image_size = image_size
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        self.kid_diffusion_steps = kid_diffusion_steps
        self.kid_image_size = kid_image_size
        self.ema = ema
        self.normalizer = layers.Normalization()
        self.network = TransformerUNet(image_size, widths, block_depth).network
        self.ema_network = TransformerUNet(image_size, widths, block_depth).network

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="noise_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="image_loss")
        self.kid = KID(name="kid", image_size=self.kid_image_size)

    def summary(self):
        print(self.network.summary())

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]


    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)


    def diffusion_schedule(self, diffusion_times):
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        return noise_rates, signal_rates


    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images


    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise

        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images,noise_rates,signal_rates,training=False)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule( next_diffusion_times)
            next_noisy_images = (next_signal_rates * pred_images + next_noise_rates * pred_noises)

        return pred_images


    def generate(self, diffusion_steps, noise):
        generated_images = self.reverse_diffusion(noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images


    def train_step(self, images):
        noises = tf.random.normal(shape=(images.shape[0], self.image_size, self.image_size, 3))
        images = self.normalizer(images, training=False)
        batch_size = images.shape[0]
        diffusion_times = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(images, pred_images)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {m.name: m.result() for m in self.metrics[:-1]}


    def test_step(self, images):
        images = self.normalizer(images, training=False)
        batch_size = images.shape[0]
        noises = tf.random.normal(shape=(batch_size, self.image_size, self.image_size, 3))##################

        diffusion_times = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=False)

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        images = self.denormalize(images)
        generated_images = self.generate(self.inference_diffusion_steps, noises)
        self.kid.update_state(images, generated_images)
        return {m.name: m.result() for m in self.metrics}


    def generate_images(self, num_images=1, steps=20):
        noises = tf.random.normal(shape=(num_images, self.image_size, self.image_size, 3))
        generated_images = self.generate(steps, noises)
        return generated_images
    

    def plot_images(self, epoch=None, logs=None, num_rows=2, num_cols=8, steps=20):
        generated_images = self.generate_images(num_rows*num_cols, steps)
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))

        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")

        plt.tight_layout()
        plt.show()
        plt.close()
        return