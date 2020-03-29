import random
import cv2
import numpy as np
from Augmentor.Operations import Operation
from PIL import Image


class Colorize(Operation):
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

        def do(image):
            t_image = None
            flags = set([eval('cv2.{}'.format(i)) for i in dir(cv2) if i.startswith('COLOR_')])
            flags = [f for f in flags if
                     f not in list(range(6, 32)) + list(range(36, 40)) + [44, 45] + list(range(50, 58)) + list(
                         range(74, 80)) + list(range(82, 86)) + list(range(127, 135))]

            while t_image is None:
                f = random.choice(flags)
                if isinstance(image, Image.Image):
                    image = np.array(image)
                try:
                    t_image = cv2.cvtColor(image, f)

                    if t_image.shape != image.shape:
                        t_image = cv2.resize(t_image, image.shape)

                except:
                    t_image = None
            return Image.fromarray(t_image)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images
