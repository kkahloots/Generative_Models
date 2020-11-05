import os
import numpy as np
import pickle
import lmdb

def get_label_by_filename(img_path):
    name, _ = os.path.splitext(img_path)
    vid_img_arr = name.split(sep=os.sep)[-1:]
    return {'label': np.array(vid_img_arr)}


def read_lmdb(lmdb_dir, num_images):
    images, labels = [], {}
    env = lmdb.open(lmdb_dir, readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))

            dataset = pickle.loads(data)
            images.append(dataset.get_image())
            labels_list = [attr for attr in dir(dataset) if
                           not callable(getattr(dataset, attr)) and (not attr.startswith("__")) and
                           (not attr in ['image', 'channels', 'size'])]

            for label in labels_list:
                # _lab = {label: eval(f'dataset.{label}')}
                # labels = {**labels, **_lab}
                if label in labels:
                    labels[label].append(eval(f'dataset.{label}'))
                else:
                    labels = {label: [eval(f'dataset.{label}')] }


    env.close()
    return {'images': images, **labels}

def read_lmdb(lmdb_dir, num_images):
    images, labels = [], {}
    env = lmdb.open(lmdb_dir, readonly=True)

    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))

            dataset = pickle.loads(data)
            images.append(dataset.get_image())
            labels_list = [attr for attr in dir(dataset) if
                           not callable(getattr(dataset, attr)) and (not attr.startswith("__")) and
                           (not attr in ['image', 'channels', 'size'])]

            for label in labels_list:
                # _lab = {label: eval(f'dataset.{label}')}
                # labels = {**labels, **_lab}
                if label in labels:
                    labels[label].append(eval(f'dataset.{label}'))
                else:
                    labels = {label: [eval(f'dataset.{label}')] }


    env.close()
    return {'images': images, **labels}