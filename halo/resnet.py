from keras import layers

class ResidualBlock(layers.Layer):
    def __init__(self, width):
        super().__init__()
        """
        Residual blocks
        """
        self.width = width
        self.convolution_1 = layers.Conv2D(width, kernel_size=1)
        self.convolution_2 = layers.Conv2D(width, kernel_size=3,padding="same")
        self.swish = layers.Activation("swish")
        self.convolution_3 = layers.Conv2D(width, kernel_size=3, padding="same")
        self.norm = layers.GroupNormalization(groups=2,epsilon=1e-5, center=False, scale=False)

    def call (self, x):
        input_width = x.shape[3]
        if input_width == self.width:
            residual = x 
        else:
            residual = self.convolution_1(x)
        x = self.norm(x)
        x = self.swish(x)
        x = self.convolution_2(x)
        x = self.swish(x)
        x = self.convolution_3(x)
        return x + residual