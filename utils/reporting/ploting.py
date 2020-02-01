import glob
import os

import IPython
import imageio
import matplotlib.pyplot as plt
from IPython import display


def plot_and_save_generated(generated, epoch, path, gray=True, save=True):
    #n_generated = int(np.sqrt(int(generated.shape[0])))
    generated = generated[:36]
    fig = plt.figure(figsize=(6, 6))
    for i in range(generated.shape[0]):
        plt.subplot(6, 6, i+1)
        if gray:
            plt.imshow(generated[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(generated[i, :, :, 0])
        plt.axis('off')

    if save:
        # tight_layout minimizes the overlap between 2 sub-plots
        fig_name = os.path.join(path, 'image_at_epoch_{:05d}.png'.format(epoch))
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