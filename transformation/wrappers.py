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
            exec(f'self.{k}={str(val)}')

    def get_image(self):
        """ Returns the image as a numpy array. """
        images = np.frombuffer(self.image, dtype=np.float32) #pay attention if you  don't use create_image_lists
        return images.reshape(*self.size, self.channels)     #then dtype will be different