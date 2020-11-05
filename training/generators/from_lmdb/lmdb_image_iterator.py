import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import Iterator, img_to_array, array_to_img
from tensorflow.keras import backend as K
import logging
from utils.reporting.logging import log_message
from utils.data_and_files.file_utils import get_file_path
import lmdb
import pickle
from utils.data_and_files.data_utils import convert_img

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
                 class_mode='categorical',
                 save_prefix='',
                 save_format='jpeg'
                 ):

        self.category = category
        self.batch_size = batch_size

        self.lmdb_dir = lmdb_dir
        self.episode_len = episode_len
        self.episode_shift = episode_shift
        self.class_mode = class_mode

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        print("Initializing Iterator " + category + " Number of images " + str(num_images))
        #print(category, lmdb_dir, batch_size, shuffle, seed)
        self.env = lmdb.open(lmdb_dir, readonly=True)

        Iterator.__init__(self, num_images, batch_size, shuffle, seed)

    def __del__(self):
        self.env.close()

    def _get_batches_of_transformed_samples(self, index_array):
        #print(index_array)
        images, labels = {}, {}
        image_for_source, image_for_target=[], []

        if len(index_array) < self.batch_size:
            diff = self.batch_size // len(index_array) + 1
            index_array = np.repeat(index_array, diff, axis=0)[:self.batch_size]

        with self.env.begin() as txn:
            if self.class_mode == 'episode':
                batch_x = np.zeros((len(index_array), self.episode_len) + (53, 70, 3), dtype=np.float32) #self.image_shape
                batch_gt = np.zeros((len(index_array), self.episode_len) + (53, 70, 3), dtype=np.float32)

                # build batch of image data
                for i, j in enumerate(index_array):
                    data = txn.get(f"{j:08}".encode("ascii"))
                    start_frame = pickle.loads(data)

                    last_frame_idx =j + self.episode_len + self.episode_shift
                    try:
                        last_frame_data = txn.get(f"{last_frame_idx:08}".encode("ascii"))
                    except:
                        j = j - (self.episode_len + self.episode_shift)

                    last_frame = pickle.loads(last_frame_data)
                    if last_frame.dir != start_frame.dir:
                        j = j - (self.episode_len + self.episode_shift)

                    imgs = []
                    for ix in range(j, j + self.episode_len):
                        img_data = txn.get(f"{ix:08}".encode("ascii"))
                        frame = pickle.loads(img_data)

                        imgs += [frame.get_image()]
                    imgs = np.array(imgs)
                    batch_x[i] = imgs

                    imgs = []
                    for ix in range(j + self.episode_shift, j + self.episode_len + self.episode_shift):
                        img_data = txn.get(f"{ix:08}".encode("ascii"))
                        frame = pickle.loads(img_data)

                        imgs += [frame.get_image()]
                    imgs = np.array(imgs)
                    batch_gt[i] = imgs
                return {"xt0": batch_x, "xt1": batch_gt}

            elif self.class_mode == 'episode_flat':
                if self.class_mode == 'episode':
                    batch_x = np.zeros((len(index_array), self.episode_len) + self.image_shape, dtype=self.dtype)
                    batch_gt = np.zeros((len(index_array), self.episode_len) + self.image_shape, dtype=self.dtype)

                    # build batch of image data
                    for i, j in enumerate(index_array):
                        data = txn.get(f"{j:08}".encode("ascii"))
                        start_frame = pickle.loads(data)

                        last_frame_idx = j + self.episode_len + self.episode_shift
                        try:
                            last_frame_data = txn.get(f"{last_frame_idx:08}".encode("ascii"))
                        except:
                            j = j - (self.episode_len + self.episode_shift)

                        last_frame = pickle.loads(last_frame_data)
                        if last_frame.dir != start_frame.dir:
                            j = j - (self.episode_len + self.episode_shift)

                        imgs = []
                        for ix in range(j, j + self.episode_len):
                            img_data = txn.get(f"{ix:08}".encode("ascii"))
                            frame = pickle.loads(img_data)

                            imgs += [frame.get_image()]
                        imgs = np.array(imgs)
                        batch_x[i] = imgs

                        imgs = []
                        for ix in range(j + self.episode_shift, j + self.episode_len + self.episode_shift):
                            img_data = txn.get(f"{ix:08}".encode("ascii"))
                            frame = pickle.loads(img_data)

                            imgs += [frame.get_image()]
                        imgs = np.array(imgs)
                        batch_gt[i] = imgs

                return {"xt0": np.reshape(batch_x, (-1,) + self.image_shape), "xt1": np.reshape(batch_gt, (-1,) + self.image_shape)}
            elif self.class_mode == 'categorical':
                for image_id in index_array:
                    data = txn.get(f"{image_id:08}".encode("ascii"))
                    dataset = pickle.loads(data)
                    image_for_source.append(dataset.get_image_for_source())
                    image_for_target.append(dataset.get_image_for_target())
                    labels_list = [attr for attr in dir(dataset) if
                                   not callable(getattr(dataset, attr)) and (not attr.startswith("__")) and
                                   (not attr in ['image', 'channels', 'size'])]

                    for label in labels_list:

                        if label in labels.keys():
                            labels[label].append(eval(f'dataset.{label}'))
                        else:
                            labels.update({label: [eval(f'dataset.{label}')]})

                images={'image_source':image_for_source, 'image_target':image_for_target}
                return {**images, **labels}

            elif self.class_mode == 'sr':
                for image_id in index_array:
                    data = txn.get(f"{image_id:08}".encode("ascii"))
                    dataset = pickle.loads(data)

                    source = dataset.get_image_for_source()
                    target = dataset.get_image_for_target()

                    image_for_source.append(source)
                    image_for_target.append(target)
                    labels_list = [attr for attr in dir(dataset) if
                                   not callable(getattr(dataset, attr)) and (not attr.startswith("__")) and
                                   (not attr in ['image_target', 'image_source', 'channels', 'size'])]

                    for label in labels_list:

                        if label in labels.keys():
                            labels[label].append(eval(f'dataset.{label}'))
                        else:
                            labels.update({label: [eval(f'dataset.{label}')]})

                #print('labels', labels.keys())
                images = {'image_source': image_for_source, 'image_target': image_for_target,  **labels}
                return images #labels#{'image_source': image_for_source, 'image_target': image_for_target}#{**images,}