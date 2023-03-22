import tensorflow as tf
import matplotlib.pyplot as plt

from halo.model import Halo
from tensorflow import keras
from halo.dataloader import Dataset


def tune(
        data_directory, 
        checkpoint_path,
        epochs=10,
        repetition=1, 
        validation_split=0.1, 
        test_split=0.2,
        min_signal_rate = 0.02,
        max_signal_rate = 0.95,
        widths = [64, 128, 192, 256],
        block_depth = 4,
        batch_size = 32,
        ema = 0.999
        ):
    
    # Kernel Inception Distance
    image_size=128
    kid_image_size = 75
    kid_diffusion_steps = 5

    # optimization
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Load dataset
    dataset = Dataset(data_directory, 
                      image_size, 
                      batch_size,
                      repetition, 
                      validation_split, 
                      test_split)
    
    # Load model
    model = Halo(image_size, 
                 widths, 
                 block_depth,
                 min_signal_rate,
                 max_signal_rate,
                 kid_diffusion_steps,
                 kid_image_size,
                 ema)
    
    model.compile(optimizer=keras.optimizers.experimental.AdamW(learning_rate=learning_rate,weight_decay=weight_decay),
                  loss=keras.losses.mean_absolute_error,
                  run_eagerly=True)
    
    model.load_weights("checkpoints/image_generator")
    model.normalizer.adapt(dataset.train)
    
    sampling_callback = keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                  filepath=checkpoint_path,
                                  save_weights_only=True,
                                  monitor="val_image_loss",
                                  mode="min",
                                  save_best_only=True)

    # Training
    model.fit(x=dataset.train,
          epochs=epochs,
          batch_size = batch_size,
          validation_data=dataset.validation,
          callbacks=[checkpoint_callback, sampling_callback])
    
    print("Halo successfully tuned on your dataset")
    print("Here are some samples")
    return model


def generate(model, n_samples, path, extension=".jpg"):
    print("generating samples")
    images = model.sample(n_samples)
    for file, image in enumerate(images):
        tf.keras.utils.save_img(path+"/"+str(file)+extension, image)
    print("done!")
    return
