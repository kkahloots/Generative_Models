import os
import pickle
import numpy as np
import json

from tqdm import tqdm
import lmdb
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from transformation.file_image_generator import create_image_lists
from transformation.file_utils import get_file_path, create_if_not_exist
from transformation.wrappers import SRDatasetWrapper
from transformation.wrappers import DatasetWrapper
from transformation.data_utils import NumpyEncoder


class LmdbTransformer:
    def __init__(self, validation_pct, valid_image_formats, image_dir=None, data_format=None, scalar=255.0):
        if image_dir is not None:
            self.image_lists = create_image_lists(image_dir, validation_pct, valid_image_formats, verbose=0)
        self.scaler = scalar
        if data_format is None:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

    def store_single_lmdb(self, filename, img, index, labels_dict, num_images):
        """ Stores a wrapper to LMDB.
        """
        map_size = num_images * img.nbytes * 5
        env = lmdb.open(filename, map_size=map_size)

        # Same as before — but let's write all the images in a single transaction
        with env.begin(write=True) as txn:
            # All key-value pairs need to be Strings
            value = DatasetWrapper(img, labels_dict)
            key = f"{index:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))

        env.close()

    def transform_store_from_numpy(self, images, labels_values, labels_names, labels_classes=None, lmdb_dir='.data/',
                                   category='training',
                                   total_number_imgs=0, file_idx=None):

        create_if_not_exist(lmdb_dir)
        num_images = images.shape[0]
        lmdb_name = lmdb_dir + os.sep + '_{}'.format(category)
        if file_idx is None:
            index = 0
        else:
            index = file_idx * 10000

        if labels_classes is None:
            for idx, (image, latents_val) in tqdm(enumerate(zip(images, labels_values)), total=num_images):
                img = np.float32(image) / self.scaler

                labels_dict = {}
                for i, A in enumerate(labels_names):
                    labels_dict[A] = latents_val[i]

                self.store_single_lmdb(index=index, filename=lmdb_name, img=img, labels_dict=labels_dict,
                                       num_images=total_number_imgs)
                index = index + 1
        else:
            for idx, (image, latents_val, labels_class) in tqdm(enumerate(zip(images, labels_values, labels_classes)),
                                                                total=num_images):
                img = np.float32(image) / self.scaler

                labels_dict = {}
                for i, A in enumerate(labels_names):
                    labels_dict[f'{A}_value'] = latents_val[i]
                    labels_dict[f'{A}_class'] = labels_class[i]

                self.store_single_lmdb(index=index, filename=lmdb_name, img=img, labels_dict=labels_dict,
                                       num_images=total_number_imgs)
                index = index + 1

    def transform_store(self, image_dir, labels_fn,
                        lmdb_dir='.data/', category='training', target_size=None,
                        color_mode='rgb'):
        create_if_not_exist(lmdb_dir)

        classes = list(self.image_lists.keys())

        total_number_of_img = 0
        for label_name in classes:
            total_number_of_img += len(self.image_lists[label_name][category])
        print('Total number of imgs for catagory ' + str(total_number_of_img))
        num_class = len(classes)
        class2id = dict(zip(classes, range(len(classes))))
        id2class = dict((v, k) for k, v in class2id.items())

        lmdb_index = 0
        for label_name in classes:
            num_images = len(self.image_lists[label_name][category])
            print('Storing ' + str(num_images) + lmdb_dir + os.sep + ' into _{} from folder {}'.format(category,
                                                                                                       label_name))

            for index, _ in enumerate(self.image_lists[label_name][category]):
                img_path = get_file_path(self.image_lists,
                                         label_name,
                                         index,
                                         image_dir,
                                         category)

                img = img_to_array(
                    load_img(
                        img_path,
                        grayscale=color_mode == 'grayscale',
                        target_size=[*reversed(target_size)],
                    ), data_format=self.data_format
                ) / self.scaler
                label_dict = labels_fn(img_path)
                name = lmdb_dir + os.sep + '_{}'.format(category)

                self.store_single_lmdb(index=lmdb_index, filename=name, img=img, labels_dict=label_dict,
                                       num_images=total_number_of_img)
                lmdb_index += 1

    def save_metadata(self, file, info_dict):
        info = json.dumps(info_dict, cls=NumpyEncoder)
        file = file + os.sep + 'meta_info.json'
        with open(file, 'w') as outfile:
            json.dump(info, outfile)

    def get_metadata(self, file):
        file = file + os.sep + 'meta_info.json'
        with open(file) as json_file:
            data = json.load(json_file)
            data_dict = json.loads(data)
            return data_dict


class SRLmdbTransformer:
    def __init__(self, validation_pct, valid_image_formats, trans_func, image_dir=None, data_format=None, scalar=255.0):
        if image_dir is not None:
            self.image_lists = create_image_lists(image_dir, validation_pct, valid_image_formats, verbose=0)
        self.scaler = scalar
        self.trans_func = trans_func
        if data_format is None:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

    def store_single_lmdb(self, filename, img, index, labels_dict, num_images):
        """
        Stores a wrapper to LMDB.
        """
        # shrink_fn = lambda image: tf.image.resize(image, inputs_shape[:-1]).numpy()

        map_size = num_images * img.nbytes * 10
        env = lmdb.open(filename, map_size=map_size)

        # Same as before — but let's write all the images in a single transaction
        with env.begin(write=True) as txn:
            # All key-value pairs need to be Strings
            if self.trans_func:
                image_target = self.trans_func(img)
            else:
                image_target = img

            value = SRDatasetWrapper(image_source=img, image_target=image_target, labels_dict=labels_dict)
            key = f"{index:08}"

            txn.put(key.encode("ascii"), pickle.dumps(value))

        env.close()

    def transform_store_from_numpy(self, images, labels_values, labels_names=None, labels_classes=None, lmdb_dir='.data/',
                                   category='training',
                                   total_number_imgs=0, file_idx=None):

        create_if_not_exist(lmdb_dir)
        num_images = images.shape[0]
        lmdb_name = lmdb_dir + os.sep + '_{}'.format(category)
        if file_idx is None:
            index = 0
        else:
            index = file_idx * 10000
        # print('Storing ' + str(num_images) + lmdb_dir + '_{}'.format(category))
        if labels_classes is None:
            for idx, (image, latents_val) in tqdm(enumerate(zip(images, labels_values)), total=num_images):
                img = np.float32(image) / self.scaler

                labels_dict = {}
                for i, A in enumerate(labels_names):
                    labels_dict[A] = latents_val[i]

                self.store_single_lmdb(index=index, filename=lmdb_name, img=img, labels_dict=labels_dict,
                                       num_images=total_number_imgs)
                index = index + 1
        else:
            for idx, (image, latents_val, labels_class) in tqdm(enumerate(zip(images, labels_values, labels_classes)),
                                                                total=num_images):
                img = np.float32(image) / self.scaler

                labels_dict = {}
                if labels_names:
                    for i, A in enumerate(labels_names):
                        labels_dict[f'{A}_value'] = latents_val[i]
                        labels_dict[f'{A}_class'] = labels_class[i]


                self.store_single_lmdb(index=index, filename=lmdb_name, img=img, labels_dict=labels_dict,
                                       num_images=total_number_imgs)
                index = index + 1

    def transform_store(self, image_dir, labels_fn,
                        lmdb_dir='.data/', category='training', target_size=None,
                        color_mode='rgb'):
        create_if_not_exist(lmdb_dir)

        classes = list(self.image_lists.keys())

        total_number_of_img = 0
        for label_name in classes:
            total_number_of_img += len(self.image_lists[label_name][category])
        print('Total number of imgs for catagory ' + str(total_number_of_img))
        num_class = len(classes)
        class2id = dict(zip(classes, range(len(classes))))
        id2class = dict((v, k) for k, v in class2id.items())

        lmdb_index = 0
        for label_name in classes:
            num_images = len(self.image_lists[label_name][category])
            print('Storing ' + str(num_images) + ' ' + lmdb_dir + os.sep + ' into _{} from folder {}'.format(category,
                                                                                                             label_name))

            for index, _ in enumerate(self.image_lists[label_name][category]):
                img_path = get_file_path(self.image_lists,
                                         label_name,
                                         index,
                                         image_dir,
                                         category)

                img = img_to_array(
                    load_img(
                        img_path,
                        grayscale=color_mode == 'grayscale',
                        target_size=[*reversed(target_size)],
                    ), data_format=self.data_format
                ) / self.scaler

                label_dict = labels_fn(img_path)
                name = lmdb_dir + os.sep + '_{}'.format(category)


                self.store_single_lmdb(index=lmdb_index, filename=name, img=img, labels_dict=label_dict,
                                       num_images=total_number_of_img)
                lmdb_index += 1

    def save_metadata(self, file, info_dict):
        info = json.dumps(info_dict, cls=NumpyEncoder)
        file = file + os.sep + 'meta_info.json'
        with open(file, 'w') as outfile:
            json.dump(info, outfile)

    def get_metadata(self, file):
        file = file + os.sep + 'meta_info.json'
        with open(file) as json_file:
            data = json.load(json_file)
            data_dict = json.loads(data)
            return data_dict
