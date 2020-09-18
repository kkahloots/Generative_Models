import hashlib
import logging
import os
import re
import warnings
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

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
        tr_lmdb_dir,
        tr_num_images,
        batch_size,
        episode_len=None,
        episode_shift=None
):
    train_datagen = LMDB_ImageIterator()

    valid_datagen = LMDB_ImageIterator()

    train_generator = train_datagen.flow_from_lmdb_lists(
        num_images=tr_num_images,
        category='training',
        lmdb_dir=tr_lmdb_dir,
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
        val_num_images,
        tr_num_images,
        batch_size,
        output_types,
        output_shapes,
        episode_len=None,
        episode_shift=None
):
    training_gen, val_gen= create_generators(val_lmdb_dir=val_lmdb_dir,
                                            val_num_images=val_num_images,
                                            tr_lmdb_dir=tr_lmdb_dir,
                                            tr_num_images=tr_num_images,
                                            batch_size=batch_size,
                                            episode_len=episode_len,
                                            episode_shift=episode_shift)


    train_generator = tf.data.Dataset.from_generator(
        lambda: training_gen,
        output_types=output_types ,
        output_shapes=output_shapes
    )

    val_generator = tf.data.Dataset.from_generator(
        lambda: training_gen,
        output_types=output_types ,
        output_shapes=output_shapes
    )
    return train_generator, val_generator