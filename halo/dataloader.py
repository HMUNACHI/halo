import pathlib
import tensorflow as tf
from functools import partial


class Dataset():

    def __init__(self,
                 data_directory, 
                 image_size, 
                 batch_size, 
                 repetition=1, 
                 validation_split=0.1, 
                 test_split=0.2
                 ):
        """
        Loads images from a directory, splits, and preprocesses and batches them
        """
        self.image_size = image_size
        self.batch_size = batch_size 
        self.repetition = repetition
        self.validation_split = validation_split
        self.test_split = test_split
        self.train, self.validation, self.test = self.__load_image_dataset(data_directory)
        

    def __load_image_dataset(self, data_directory):
        # Parse and load the images from directory
        image_paths = []
        extensions = ('*/*.jpg', '*/*.jpeg', '*/*png')

        for extension in extensions:
            paths = pathlib.Path(data_directory).glob(extension)
            image_paths.extend([str(path) for path in paths])

        image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
        validation_split_idx = int(len(image_paths) * self.validation_split)
        test_split_idx = int(len(image_paths) * (self.test_split + self.validation_split))
        
        # Train pipeline
        train_dataset = (tf.data.Dataset.from_tensor_slices(image_paths[test_split_idx:])
                        .map(self.__preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                        .cache()
                        .shuffle(10 * self.batch_size)
                        .repeat(self.repetition)
                        .batch(self.batch_size, drop_remainder=True)
                        .prefetch(buffer_size=tf.data.AUTOTUNE))
        
        # Validation pipeline
        validation_dataset = (tf.data.Dataset.from_tensor_slices(image_paths[:validation_split_idx])
                            .map(self.__preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                            .cache()
                            .shuffle(10 * self.batch_size)
                            .repeat(self.repetition)
                            .batch(self.batch_size, drop_remainder=True)
                            .prefetch(buffer_size=tf.data.AUTOTUNE))
        
        # Test pipeline
        test_dataset = (tf.data.Dataset.from_tensor_slices(image_paths[validation_split_idx:test_split_idx])
                            .map(self.__preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
                            .cache()
                            .shuffle(10 * self.batch_size)
                            .repeat(self.repetition)
                            .batch(self.batch_size, drop_remainder=True)
                            .prefetch(buffer_size=tf.data.AUTOTUNE))
        
        return train_dataset, validation_dataset, test_dataset

    
    @tf.function
    def __preprocess_image(self, image_path):
        # Read image files
        raw_image = tf.io.read_file(image_path)

        try:
            image = tf.io.decode_jpeg(raw_image, channels=3)
        except:
            image = tf.image.decode_png(raw_image, channels=3)

        # Centre crop image
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        crop_size = tf.minimum(height, width)
        image = tf.image.crop_to_bounding_box(image,
                                            (height - crop_size) // 2,
                                            (width - crop_size) // 2,
                                            crop_size,
                                            crop_size)
        
        # Resize and clip
        image = tf.image.resize(image, size=[self.image_size, self.image_size], antialias=True)
        return tf.clip_by_value(image / 255.0, 0.0, 1.0)