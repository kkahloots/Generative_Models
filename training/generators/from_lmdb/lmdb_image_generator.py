import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformation.lmdb_transformer import LmdbTransformer


from utils.data_and_files.data_utils import infer_type
from training.generators.from_lmdb.lmdb_image_iterator import LMDB_ImageIterator

class LMDB_ImageGenerator(ImageDataGenerator):
    def flow_from_lmdb_lists(self,
                             num_images,
                             category,
                             lmdb_dir,
                             batch_size,
                             episode_len=None,
                             episode_shift=None,
                             shuffle=True,
                             class_mode='categorical',
                             seed=None
                             ):
        return LMDB_ImageIterator(
            num_images=num_images,
            category=category,
            lmdb_dir=lmdb_dir,
            batch_size=batch_size,
            episode_len=episode_len,
            episode_shift=episode_shift,
            class_mode = class_mode,
            shuffle=shuffle,
            seed=seed)


def create_generators(
        val_lmdb_dir,
        val_num_images,
        tra_lmdb_dir,
        tra_num_images,
        batch_size,
        episode_len=None,
        episode_shift=None,
        class_mode='categorical'
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
        class_mode=class_mode,
        seed=0)

    validation_generator = valid_datagen.flow_from_lmdb_lists(
        num_images=val_num_images,
        category='validation',
        lmdb_dir=val_lmdb_dir,
        batch_size=batch_size,
        episode_len=episode_len,
        episode_shift=episode_shift,
        class_mode=class_mode,
        seed=0)

    return train_generator, validation_generator


def get_generators(
        lmdb_dir,
        batch_size,
        episode_len=None,
        episode_shift=None,
        class_mode='categorical',
        return_itr=False
):
    transformer = LmdbTransformer(image_dir=lmdb_dir,
                                  validation_pct=20,
                                  valid_image_formats='png')
    meta = transformer.get_metadata(lmdb_dir)

    training_gen, val_gen = create_generators(val_lmdb_dir=os.path.join(lmdb_dir, '_validation'),
                                             val_num_images=meta['val_num_images'],
                                             tra_lmdb_dir=os.path.join(lmdb_dir, '_training'),
                                             tra_num_images=meta['tra_num_images'],
                                             batch_size=batch_size,
                                             episode_len=episode_len,
                                             episode_shift=episode_shift,
                                             class_mode = class_mode
                                             )


    data = training_gen.next()
    dtypes = {k: infer_type(v[0]) for k, v in data.items()}

    if return_itr:
        return training_gen, val_gen

    else:
        train_generator = tf.data.Dataset.from_generator(
            lambda: training_gen,
            output_types=dtypes,
        )

        val_generator = tf.data.Dataset.from_generator(
            lambda: val_gen,
            output_types=dtypes,
        )
    return  train_generator, val_generator



