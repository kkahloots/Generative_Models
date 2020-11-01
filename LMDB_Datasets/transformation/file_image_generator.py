import glob
import hashlib
import logging
import os
import re
import warnings


from LMDB_Datasets.transformation.data_utils import as_bytes
from LMDB_Datasets.transformation.logging import log_message

def create_image_lists(image_dir, validation_pct, valid_image_formats, max_num_images_per_class=2 ** 27 - 1, verbose = 1):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    # Arguments
        image_dir: string path to a folder containing subfolders of images.
        validation_pct: integer percentage of images reserved for validation.

    # Returns
        dictionary of label subfolder, with images split into training
        and validation sets within each label.
    """
    if not os.path.isdir(image_dir):
        raise ValueError("Image directory {} not found.".format(image_dir))
    image_lists = {}
    sub_dirs = [x[0] for x in os.walk(image_dir)]

    sub_dirs_without_root = sub_dirs[1:]  # first element is root directory
    for sub_dir in sub_dirs_without_root:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        if verbose == 1:
            log_message("Looking for images in '{}'".format(dir_name), logging.DEBUG)

        if isinstance(valid_image_formats, str):
            valid_image_formats = [valid_image_formats]

        for extension in valid_image_formats:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            msg = 'No files found'
            if verbose == 1:
                log_message(msg, logging.WARN)
            warnings.warn(msg)
            continue
        else:
            if verbose == 1:
                log_message('{} file found'.format(len(file_list)), logging.INFO)
        if len(file_list) < 20:
            msg = 'Folder has less than 20 images, which may cause issues.'
            if verbose == 1:
                log_message(msg, logging.WARN)
            warnings.warn(msg)
        elif len(file_list) > max_num_images_per_class:
            msg='WARNING: Folder {} has more than {} images. Some '\
                          'images will never be selected.' \
                          .format(dir_name, max_num_images_per_class)
            log_message(msg, logging.WARN)
            warnings.warn(msg)
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # Get the hash of the file name and perform variant assignment.
            hash_name = hashlib.sha1(as_bytes(base_name)).hexdigest()
            hash_pct = ((int(hash_name, 16) % (max_num_images_per_class  + 1)) *
                        (100.0 / max_num_images_per_class))
            if hash_pct < validation_pct:
                validation_images.append(base_name)
            else:
                training_images.append(base_name)
        image_lists[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'validation': validation_images,
        }
    return image_lists
