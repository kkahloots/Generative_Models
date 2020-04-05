import os
import random
import string
import csv
import logging
from utils.reporting.logging import log_message

def create_random_string(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def get_file_path(file_lists, label_name, index, example_dir, category):
    """"Returns a path to a file for a label at the given index.

    # Arguments
      file_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      file_dir: Root folder string of the subfolders containing the training
      example.
      category: Name string of set to pull files from - training, testing, or
      validation.

    # Returns
      File system path string to an image that meets the requested parameters.
    """
    if label_name not in file_lists:
        raise ValueError('Label does not exist ', label_name)

    label_lists = file_lists[label_name]
    if category not in label_lists:
        raise ValueError('Category does not exist ', category)

    category_list = label_lists[category]
    if not category_list:
        raise ValueError('Label %s has no images in the category %s.',
                         label_name, category)

    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(example_dir, sub_dir, base_name)
    return full_path

def create_if_not_exist(directories):
    """
    dirs - a list of directories to create if these directories are not found
    :param directories:
    """
    if not isinstance(directories, list):
        directories = [directories]

    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)

        except Exception as err:
            log_message("Creating directory {},  error: {}".format(directory, err), logging.ERROR)

def log(file_name, message):
    header = file_name.split(os.path.sep)[-1].split('_')[0]
    fieldnames = [*message]
    with open(file_name+'.csv', mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if message['Epoch']==0:
            writer.writeheader()
        writer.writerow(message)

def inspect_log(file_name):
    try:
        with open(file_name, 'r', encoding="utf-8", errors="ignore") as scraped:
            reader = csv.reader(scraped, delimiter=',')
            last_row = [0]
            for row in reader:
                if row:  # avoid blank lines
                    last_row = row
        if last_row[0]==0:
            log_message('No former training found ... ', logging.ERROR)
        else:
            log_message('Found Record for {} Epochs'.format(int(last_row[0])), logging.INFO)
        return int(last_row[0])
    except:
        log_message('No former training found ... ', logging.ERROR)
        return 0

