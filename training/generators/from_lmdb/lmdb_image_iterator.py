import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import Iterator, load_img, img_to_array, array_to_img
from keras import backend as K
import logging
from utils.reporting.logging import log_message
from utils.data_and_files.file_utils import get_file_path
import lmdb
import pickle

class LMDB_ImageIterator(Iterator):

    def __init__(self,
                 num_images,
                 category,
                 lmdb_dir,
                 batch_size,
                 episode_len=20,
                 episode_shift=10,
                 shuffle=True,
                 seed=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='jpeg'
                 ):

        self.category = category
        self.batch_size = batch_size

        self.lmdb_dir = lmdb_dir
        self.episode_len = episode_len
        self.episode_shift = episode_shift

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        print("Initializing Iterator " + category + " Number of images " + str(num_images))
        print(category, lmdb_dir, batch_size, shuffle, seed)
        self.env = lmdb.open(lmdb_dir, readonly=True)

        Iterator.__init__(self, num_images, batch_size, shuffle, seed)

    def __del__(self):
        self.env.close()

    def _get_batches_of_transformed_samples(self, index_array):
        print(index_array)
        images, labels = [], {}

        if len(index_array) < self.batch_size:
            diff = self.batch_size // len(index_array) + 1
            index_array = np.repeat(index_array, diff, axis=0)[:self.batch_size]

        else:
            with self.env.begin() as txn:
                for image_id in index_array:
                    data = txn.get(f"{image_id:08}".encode("ascii"))
                    dataset = pickle.loads(data)
                    images.append(dataset.get_image())
                    labels_list = [attr for attr in dir(dataset) if
                                   not callable(getattr(dataset, attr)) and (not attr.startswith("__")) and
                                   (not attr in ['image', 'channels', 'size'])]

                    for label in labels_list:

                        if label in labels.keys():
                            labels[label].append(eval(f'dataset.{label}'))
                        else:
                            labels.update({label: [eval(f'dataset.{label}')]})
        return {'images': images, **labels}