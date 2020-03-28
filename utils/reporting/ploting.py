import glob
import os

import IPython
import imageio
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image

def plot_and_save_generated(generated, epoch, path, gray=True, save=True):
    #n_generated = int(np.sqrt(int(generated.shape[0])))
    generated = generated[:36]
    fig = plt.figure(figsize=(6, 6))
    for i in range(generated.shape[0]):
        plt.subplot(6, 6, i+1)
        if gray:
            plt.imshow(generated[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(generated[i])
        plt.axis('off')

    if save:
        # tight_layout minimizes the overlap between 2 sub-plots
        fig_name = os.path.join(path, 'image_at_epoch_{:06d}.png'.format(epoch))
        plt.savefig(fig_name)
        plt.close()


def animate(model_name, path='./experiments'):
    anim_file = os.path.join(path, model_name+'.gif')

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(path, 'image*.png'))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)

    if IPython.version_info >= (6, 2, 0, ''):
        display.Image(filename=anim_file)


def append_images(images, direction='horizontal', bg_color=(255,255,255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im