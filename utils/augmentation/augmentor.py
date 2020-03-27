import numpy as np
from Augmentor.Operations import Operation, Skew, Distort, Rotate, Shear, Flip, Zoom, HistogramEqualisation
from PIL import Image
import cv2

from utils.augmentation.Cloner import Clone
from utils.augmentation.Colorizer import Colorize
from utils.augmentation.Skitcher import Skitch

import random
def do_operation(opt, image, padding=50):
    image_cv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    dim = image_cv.shape[:2]

    if str(opt)=='Skew':
        color = [0, 0, 0]
        top, bottom = padding, padding
        left, right = padding, padding
        image_cv = cv2.copyMakeBorder(image_cv, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    image_cv = opt.perform_operation([Image.fromarray(image_cv)])[0]
    image_cv = cv2.cvtColor((np.array(image_cv) * 255).astype(np.uint8), cv2.IMREAD_COLOR)

    # resize image
    image_cv = cv2.resize(image_cv, dim, interpolation=cv2.INTER_AREA)
    return np.asarray(image_cv/255.0)

operations = {0: lambda: Skew(probability=1, skew_type="RANDOM", magnitude=0.7),
              1: lambda: Distort(probability=1, grid_width=random.randint(1, 50), grid_height=random.randint(1, 50),
                                 magnitude=5),
              2: lambda: Rotate(probability=1, rotation=random.randint(1, 360)),
              3: lambda: Shear(probability=1, max_shear_left=0, max_shear_right=random.randint(5, 15)) \
                 if random.randint(0,1)==1 else Shear(probability=1, max_shear_left=random.randint(5, 15), max_shear_right=0),
              4: lambda: Zoom(probability=1, min_factor=random.randint(2, 10)/10, max_factor=random.randint(10, 12)/10),
              5: lambda: Colorize(probability=1),
              6: lambda: Skitch(probability=1),
              7: lambda: Flip(probability=1, top_bottom_left_right="RANDOM"),
              8: lambda: Clone(probability=1)
             }
