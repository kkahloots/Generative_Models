import cv2
import numpy as np
from Augmentor.Operations import Operation
import scipy.ndimage

class Skitch(Operation):
    """
    This class is used to Color images .
    """

    def __init__(self, probability):
        """
        As the aspect ratio is always kept constant, only a
        :attr:`scale_factor` is required for scaling the image.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param scale_factor: The factor by which to scale, where 1.5 would
         result in an image scaled up by 150%.
        :type probability: Float
        :type scale_factor: Float
        """
        Operation.__init__(self, probability)

    def perform_operation(self, images):
        """
        Scale the passed :attr:`images` by the factor specified during
        instantiation, returning the scaled image.
        :param images: The image to scale.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        augmented_images = []

        for image in images:
            sketch_color, sketch_gray = cv2.pencilSketch(np.array(image).astype(np.float32), sigma_s=200, sigma_r=0.05, shade_factor=0.1)
            augmented_images.append(255-sketch_color)

        return augmented_images
