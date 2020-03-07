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
        
        #== Parameters         
        BLUR = 21
        CANNY_THRESH_1 = 10
        CANNY_THRESH_2 = 50
        MASK_DILATE_ITER = 10
        MASK_ERODE_ITER = 10
        MASK_COLOR = (0.0,0.0,0.0) # In BGR format

        def do(image):
            t_image = None
            flags = [eval('cv2.{}'.format(i)) for i in dir(cv2) if i.startswith('COLOR_')]

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
	    
            gray = cv2.cvtColor(t_image,cv2.COLOR_BGR2GRAY)

            #-- Edge detection 
            edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
            edges = cv2.dilate(edges, None)
            edges = cv2.erode(edges, None)

            #-- Find contours in edges, sort by area 
            contour_info = []
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                contour_info.append((
                             c,
                             cv2.isContourConvex(c),
                             cv2.contourArea(c),
                           ))
            contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
            if contour_info == []:
                countiue
            
            max_contour = contour_info[0]

            #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
            # Mask is black, polygon is white
            mask = np.zeros(edges.shape)
            cv2.fillConvexPoly(mask, max_contour[0], (255))

            #-- Smooth mask, then blur it
            mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
            mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
            mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
            mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

            #-- Blend masked img into MASK_COLOR background
            mask_stack  = mask_stack.astype('float32') / 255.0         
            t_image         = t_image.astype('float32') / 255.0    
            masked = (mask_stack * t_image) + ((1-mask_stack) * MASK_COLOR)  
            masked = (masked * 255).astype('uint8') 
       
            if masked.shape != image.shape:
                masked = cv2.resize(masked, image.shape)         		

            return masked

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images
