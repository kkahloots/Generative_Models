import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import Iterator, load_img, img_to_array, array_to_img
from keras import backend as K
import logging
from utils.reporting.logging import log_message
from utils.data_and_files.file_utils import get_file_path

class ImageIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        image_lists: Dictionary of training images for each label.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `None`: no targets get yielded (only input images are yielded).
            `episode`: a sequence of images yielded (with time shift).
            `func`: custom transformation gets the input images and the transformed.

        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self,
                 image_lists,
                 image_data_generator,
                 category, image_dir,
                 target_size=(256, 256, 3),
                 color_mode='rgb',
                 class_mode='categorical',
                 batch_size=32,
                 episode_len=20,
                 episode_shift=10,
                 shuffle=True,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='jpeg',
                 dtype=K.floatx()
                 ):
        if data_format is None:
            data_format = K.image_data_format()

        classes = list(image_lists.keys())
        self.category = category
        self.batch_size = batch_size
        self.num_class = len(classes)
        self.image_lists = image_lists
        self.image_dir = image_dir
        self.episode_len = episode_len
        self.episode_shift = episode_shift

        how_many_files = 0
        for label_name in classes:
            for _ in self.image_lists[label_name][category]:
                how_many_files += 1

        self.samples = how_many_files
        self.class2id = dict(zip(classes, range(len(classes))))
        self.id2class = dict((v, k) for k, v in self.class2id.items())
        self.classes = np.zeros((self.samples,), dtype='int32')

        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        self.image_shape = self.target_size

        if (class_mode not in {'categorical', 'binary', 'sparse', 'episode', 'episode_flat', None}) and (not hasattr(class_mode, '__call__')):
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "episode", or None.')
        self.class_mode = class_mode
        self.dtype = dtype
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        i = 0
        self.filenames = []
        for label_name in classes:
            for j, _ in enumerate(self.image_lists[label_name][category]):
                self.classes[i] = self.class2id[label_name]
                img_path = get_file_path(self.image_lists,
                                          label_name,
                                          j,
                                          self.image_dir,
                                          self.category)
                self.filenames.append(img_path)
                i += 1
        log_message("Found {} {} files".format(len(self.filenames), category), logging.INFO)
        Iterator.__init__(self, self.samples, self.batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # with self.lock:
        #    index_array, current_index, current_batch_size = next(self.index_generator)

        # The transformation of images is not under thread lock
        # so it can be done in parallel

        if len(index_array) < self.batch_size:
            diff = self.batch_size//len(index_array) + 1
            index_array = np.repeat(index_array, diff, axis=0)[:self.batch_size]

        grayscale = self.color_mode == 'grayscale'
        if self.class_mode == 'episode':
            batch_x = np.zeros((len(index_array), self.episode_len) + self.image_shape, dtype=self.dtype)
            batch_gt = np.zeros((len(index_array), self.episode_len) + self.image_shape, dtype=self.dtype)

            def get_filename(path):
                folder, file = path.split(os.path.sep)[-2:]
                file_name = file.split('.')[0]
                return folder, int(file_name)


            sorted_filenames = sorted(self.filenames, key=lambda f: get_filename(f))

            max_ix = int(get_filename(sorted_filenames[-1])[1])
            last_ix = max_ix - (self.episode_len + self.episode_shift)

            # build batch of image data
            for i, j in enumerate(index_array):
                if j > last_ix:
                    j = j - last_ix

                imgs = []
                for ix in range(j, j+self.episode_len):
                    imgs += [
                        self.image_data_generator.standardize(
                            img_to_array(
                                load_img(
                                    sorted_filenames[ix],
                                    grayscale=grayscale,
                                    target_size=self.target_size
                                ),
                                data_format=self.data_format
                            )
                        )
                    ]
                imgs = np.array(imgs)
                batch_x[i] = imgs

                imgs = []
                for ix in range(j+self.episode_shift, j+self.episode_len+self.episode_shift):
                    imgs += [
                        self.image_data_generator.standardize(
                            img_to_array(
                                load_img(
                                    sorted_filenames[ix],
                                    grayscale=grayscale,
                                    target_size=self.target_size
                                ),
                                data_format=self.data_format
                            )
                        )
                    ]

                imgs = np.array(imgs)
                batch_gt[i] = imgs
            return batch_x, batch_gt

        elif self.class_mode == 'episode_flat':
            batch_x = np.zeros((len(index_array), self.episode_len) + self.image_shape, dtype=self.dtype)
            batch_gt = np.zeros((len(index_array), self.episode_len) + self.image_shape, dtype=self.dtype)

            def get_filename(path):
                folder, file = path.split(os.path.sep)[-2:]
                file_name = file.split('.')[0]
                return folder, int(file_name)

            sorted_filenames = sorted(self.filenames, key=lambda f: get_filename(f))

            max_ix = int(get_filename(sorted_filenames[-1])[1])
            last_ix = max_ix - (self.episode_len + self.episode_shift)

            # build batch of image data
            for i, j in enumerate(index_array):
                if j > last_ix:
                    j = j - last_ix

                imgs = []
                for ix in range(j, j + self.episode_len):
                    imgs += [
                        self.image_data_generator.standardize(
                            img_to_array(
                                load_img(
                                    sorted_filenames[ix],
                                    grayscale=grayscale,
                                    target_size=self.target_size
                                ),
                                data_format=self.data_format
                            )
                        )
                    ]
                imgs = np.array(imgs)
                batch_x[i] = imgs

                imgs = []
                for ix in range(j + self.episode_shift, j + self.episode_len + self.episode_shift):
                    imgs += [
                        self.image_data_generator.standardize(
                            img_to_array(
                                load_img(
                                    sorted_filenames[ix],
                                    grayscale=grayscale,
                                    target_size=self.target_size
                                ),
                                data_format=self.data_format
                            )
                        )
                    ]

                imgs = np.array(imgs)
                batch_gt[i] = imgs
            return np.reshape(batch_x, (-1,)+self.image_shape ), np.reshape(batch_gt, (-1,)+self.image_shape)
        else:
            batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
            # build batch of image data
            for i, j in enumerate(index_array):
                img = load_img(self.filenames[j],
                               grayscale=grayscale,
                               target_size=self.target_size)
                x = img_to_array(img, data_format=self.data_format)
                x = self.image_data_generator.random_transform(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x

            # optionally save augmented images to disk for debugging purposes
            if self.save_to_dir:
                for i, j in enumerate(index_array):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=j,
                                                                      hash=np.random.randint(10000),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
            # build batch of labels
            if self.class_mode == 'sparse':
                batch_y = self.classes[index_array]
            elif self.class_mode == 'binary':
                batch_y = self.classes[index_array].astype(K.floatx())
            elif self.class_mode == 'categorical':
                batch_y = np.zeros((len(batch_x), self.num_class),dtype=K.floatx())
                for i, label in enumerate(self.classes[index_array]):
                    batch_y[i, label] = 1.

            elif self.class_mode is None:
                return batch_x
            else:
                return batch_x, self.class_mode(batch_x)
            return batch_x, batch_y

