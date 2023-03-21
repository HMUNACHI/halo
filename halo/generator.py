from keras import layers
from tensorflow import keras
from halo.resnet import ResidualBlock
from halo.embeddings import SinusoidalEmbedding
from halo.transformer import VisionTransformerBlock

class UNet():
    def __init__(self, 
                 image_size, 
                 widths, 
                 block_depth,
                 embed_dims=64,
                 embed_min_freq=1.0,
                 embed_max_freq=1000.0):
        
        super().__init__()
        """
        Image Generation Model
        """
        self.concatenate = layers.Concatenate()
        self.convolution_1 = layers.Conv2D(widths[0], kernel_size=1)
        self.convolution_2 = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")
        self.upsampling = layers.UpSampling2D(size=image_size, interpolation="nearest")
        self.sinusoidal_embedding = SinusoidalEmbedding(embed_dims, embed_min_freq, embed_max_freq)

        self.down_blocks = [self.__DownBlock(width, block_depth) for width in widths[:-1]]
        self.residual_blocks =  [ResidualBlock(widths[-1]) for _ in range(block_depth)]
        self.up_blocks =  [self.__UpBlock(width, block_depth) for width in reversed(widths[:-1])]
        
        self.skips = []
        self.network = self.create(image_size)

        
    def __DownBlock(self, width, block_depth):
        def apply(x):
            for _ in range(block_depth):
                x = ResidualBlock(width)(x)
                #x = VisionTransformerBlock(width, num_heads=2, head_size=int(width/2))(x)
                self.skips.append(x)
            x = layers.AveragePooling2D(pool_size=2)(x)
            return x
        return apply


    def __UpBlock(self, width, block_depth):
        def apply(x):
            x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
            for _ in range(block_depth):
                x = layers.Concatenate()([x, self.skips.pop()])
                x = ResidualBlock(width)(x)
                #x = VisionTransformerBlock(width, num_heads=2, head_size=int(width/2))(x)
            return x
        return apply


    def create(self, image_size):
        noisy_images = keras.Input(shape=(image_size, image_size, 3))
        noise_variances = keras.Input(shape=(1,1,1))

        e = self.sinusoidal_embedding(noise_variances)
        e = self.upsampling(e)

        x = self.convolution_1(noisy_images)
        x = self.concatenate([x, e])

        for block in self.down_blocks:
            x = block(x)
        for block in self.residual_blocks:
            x = block(x)
        for block in self.up_blocks:
            x = block(x)

        outputs = self.convolution_2(x)

        return keras.Model([noisy_images, noise_variances], outputs)