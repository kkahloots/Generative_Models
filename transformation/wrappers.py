import numpy as np


# Wrapper class for dataset
class DatasetWrapper:
    def __init__(self, image, labels_dict):
        try:
            self.channels = image.shape[2]
        except:
            self.channels = 1
        self.size = image.shape[:2]
        self.image = image.tobytes()
        for k, val in labels_dict.items():
            exec("self.{} = '{}'".format(k, str(val)))

    def get_image(self):
        """ Returns the image as a numpy array. """
        images = np.frombuffer(self.image, dtype=np.float32)  # pay attention if you  don't use create_image_lists
        return images.reshape(*self.size, self.channels)  # then dtype will be different


# Wrapper class for dataset
class SRDatasetWrapper:
    def __init__(self, xt0, xt1, labels_dict):
        try:
            self.xt0_channels = xt0.shape[2]
            self.xt1_channels = xt1.shape[2]
        except:
            self.source_channels = 1
            self.target_channels = 1
        self.xt0_size = xt0.shape[:2]
        self.xt1_size = xt1.shape[:2]
        self.xt0 = xt0.tobytes()
        self.xt1 = xt1.tobytes()

        for k, val in labels_dict.items():
            exec("self.{} = '{}'".format(k, str(val)))

    def get_xt0(self):
        """ Returns the image as a numpy array. """
        images = np.frombuffer(self.xt0,
                               dtype=np.float32)  # pay attention if you  don't use create_image_lists
        return images.reshape(*self.xt0_size, self.xt0_channels)  # then dtype will be different

    def get_xt1(self):
        """ Returns the image as a numpy array. """
        images = np.frombuffer(self.xt1,
                               dtype=np.float32)  # pay attention if you  don't use create_image_lists
        return images.reshape(*self.xt1_size, self.xt1_channels)  # then dtype will be different

