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

        def do(image):
            def grayscale(rgb):
                return cv2.cvtColor(np.array(rgb, dtype=np.float32), cv2.COLOR_BGR2GRAY)

            def dodge(front, back):
                result = front * 255 / (255 - back)
                result[result > 255] = 255
                result[back == 255] = 255
                return result / 255.0

            gray_img = grayscale(image)
            inverted_img = 255 - gray_img

            blur_img = scipy.ndimage.filters.gaussian_filter(inverted_img, sigma=10)
            final_img = dodge(blur_img, gray_img)

            return 255-final_img

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images
