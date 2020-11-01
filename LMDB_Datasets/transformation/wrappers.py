import numpy as np

#Wrapper class for dataset
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
        images = np.frombuffer(self.image, dtype=np.float32) #pay attention if you  don't use create_image_lists
        return images.reshape(*self.size, self.channels)     #then dtype will be different


#Wrapper class for dataset
class SRDatasetWrapper:
    def __init__(self, image_source, image_target, labels_dict):
        try:
            self.source_channels = image_source.shape[2]
            self.target_channels = image_target.shape[2]
        except:
            self.source_channels = 1
            self.target_channels = 1
        self.source_size = image_source.shape[:2]
        self.target_size= image_target.shape[:2]
        self.image_source = image_source.tobytes()
        self.image_target = image_target.tobytes()

        for k, val in labels_dict.items():
            exec("self.{} = '{}'".format(k, str(val)))

    def get_image_for_source(self):
        """ Returns the image as a numpy array. """
        images = np.frombuffer(self.image_source, dtype=np.float32) #pay attention if you  don't use create_image_lists
        return images.reshape(*self.source_size, self.source_channels)     #then dtype will be different
    def get_image_for_target(self):
      """ Returns the image as a numpy array. """
      images = np.frombuffer(self.image_target, dtype=np.float32) #pay attention if you  don't use create_image_lists
      return images.reshape(*self.target_size, self.target_channels)     #then dtype will be different