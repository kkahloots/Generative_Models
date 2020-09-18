import hashlib
import logging
import os
import re
import warnings
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from transformation.lmdb_transformer import LmdbTransformer


from utils.data_and_files.data_utils import as_bytes
from utils.reporting.logging import log_message
from training.generators.from_lmdb.lmdb_image_iterator import LMDB_ImageIterator

class LMDB_ImageGenerator(ImageDataGenerator):
    def flow_from_lmdb_lists(self,
                             num_images,
                             category,
                             lmdb_dir,
                             batch_size,
                             episode_len=None,
                             episode_shift=None,
                             color_mode='rgb',
                             shuffle=True,
                             seed=None
                             ):
        return LMDB_ImageIterator(
            num_images=num_images,
            category=category,
            lmdb_dir=lmdb_dir,
            batch_size=batch_size,
            episode_len=episode_len,
            episode_shift=episode_shift,
            shuffle=shuffle,
            seed=seed)


def create_generators(
        val_lmdb_dir,
        val_num_images,
        tra_lmdb_dir,
        tra_num_images,
        batch_size,
        episode_len=None,
        episode_shift=None
):
    train_datagen = LMDB_ImageGenerator()

    valid_datagen = LMDB_ImageGenerator()

    train_generator = train_datagen.flow_from_lmdb_lists(
        num_images=tra_num_images,
        category='training',
        lmdb_dir=tra_lmdb_dir,
        batch_size=batch_size,
        episode_len=episode_len,
        episode_shift=episode_shift,
        seed=0)

    validation_generator = valid_datagen.flow_from_lmdb_lists(
        num_images=val_num_images,
        category='validation',
        lmdb_dir=val_lmdb_dir,
        batch_size=batch_size,
        episode_len=episode_len,
        episode_shift=episode_shift,
        seed=0)

    return train_generator, validation_generator


def get_generators(
        lmdb_dir,
        batch_size,
        output_types,
        output_shapes,
        episode_len=None,
        episode_shift=None,
):
    transformer = LmdbTransformer(image_dir=lmdb_dir,
                                  validation_pct=20,
                                  valid_image_formats='png')
    meta = transformer.get_metadata(lmdb_dir)

    training_gen, val_gen= create_generators(val_lmdb_dir=os.path.join(lmdb_dir, '_validation'),
                                            val_num_images=meta['val_num_images'],
                                            tra_lmdb_dir=os.path.join(lmdb_dir, '_training'),
                                            tra_num_images=meta['tra_num_images'],
                                            batch_size=batch_size,
                                            episode_len=episode_len,
                                            episode_shift=episode_shift)


    # train_generator = tf.data.Dataset.from_generator(
    #     lambda: training_gen,
    #     output_types=output_types ,
    #     output_shapes=output_shapes
    # )
    #
    # val_generator = tf.data.Dataset.from_generator(
    #     lambda: training_gen,
    #     output_types=output_types ,
    #     output_shapes=output_shapes
    # )
    return training_gen, val_gen