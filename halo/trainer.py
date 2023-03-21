import tensorflow as tf
import tensorflow_model_optimization as tfmot

from halo.model import Halo
from tensorflow import keras


####################### In Development (Do Not Use) #####################
def pretrain(
            model,
            dataset,
            checkpoint_path,
            epochs=1,
            image_size=256,
            min_signal_rate = 0.02,
            max_signal_rate = 0.95,
            widths = [64, 128, 192, 256],
            block_depth = 4,
            batch_size = 32,
            ema = 0.999
            ):
    
    kid_image_size = 75
    kid_diffusion_steps = 5
    learning_rate = 1e-3
    weight_decay = 1e-4

    model = Halo(
                 image_size, 
                 widths, 
                 block_depth,
                 min_signal_rate,
                 max_signal_rate,
                 kid_diffusion_steps,
                 kid_image_size,
                 ema
                 )
    
    model.compile(optimizer=keras.optimizers.experimental.AdamW(learning_rate=learning_rate,weight_decay=weight_decay),
                  loss=keras.losses.mean_absolute_error)
    
    model.normalizer.adapt(dataset.train)
    sampling_callback = keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                  filepath=checkpoint_path,
                                  save_weights_only=True,
                                  monitor="val_image_loss",
                                  mode="min",
                                  save_best_only=True)

    # Pre-train model
    model.fit(
          x=dataset.train,
          epochs=epochs,
          batch_size = batch_size,
          validation_data=dataset.validation,
          callbacks=[checkpoint_callback, sampling_callback]
          )
    
    # Cluster model weights
    clustered_model = cluster_weights(model, n_clusters=16)


    # Fine-tune clustered model
    clustered_model.fit(
          x=dataset.train,
          epochs=10,
          batch_size = batch_size,
          validation_data=dataset.validation,
          callbacks=[checkpoint_callback, sampling_callback]
          )
    
    # Strip and quantize model
    stripped_clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)
    quantized_model = quantize_model(stripped_clustered_model)

    # Fine-tune quantized model
    quantized_model.fit(
          x=dataset.train,
          epochs=10,
          batch_size = batch_size,
          validation_data=dataset.validation,
          callbacks=[checkpoint_callback, sampling_callback]
          )
    
    # Save model
    save_model(quantized_model, "pretrained_model.tflite")
    return 


def cluster_weights(model, n_clusters):
    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    clustering_params = {
                         'number_of_clusters': n_clusters,
                         'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
                         'cluster_per_channel': True,
                        }

    clustered_model = cluster_weights(model, **clustering_params)

    # Use smaller learning rate for fine-tuning
    optimizer = keras.optimizers.experimental.AdamW(learning_rate=1e-5,weight_decay=1e-4)
    clustered_model.compile(optimizer=optimizer,
                  loss=keras.losses.mean_absolute_error)
    
    return clustered_model

    

def quantize_model(stripped_clustered_model):
    quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
            stripped_clustered_model)
        
    quantized_model = tfmot.quantization.keras.quantize_apply(
            quant_aware_annotate_model,
            tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme())

    optimizer = keras.optimizers.experimental.AdamW(learning_rate=1e-5,weight_decay=1e-4)
    
    quantized_model.compile(optimizer=optimizer,
            loss=keras.losses.mean_absolute_error)
        
    return quantized_model


def save_model(model, file_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(file_name, 'wb') as f:
        f.write(tflite_model)

    return