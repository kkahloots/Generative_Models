from PIL import Image
import os
import numpy as np
from evaluation.generativity_metrics.shared_api import slerp

def reconstruct_from_a_batch(model, data_generator, save_dir):
    # Generate latents from the data
    original_data = next(data_generator)
    images = model.reconstruct(original_data).numpy()

    for i, (original_image, reconstructed_image) in enumerate(zip(original_data, images)):
        fig_name = os.path.join(save_dir, 'reconstructed_image_{:06d}.png'.format(i))
        image = Image.fromarray((reconstructed_image * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)

        fig_name = os.path.join(save_dir, 'original_image_{:06d}.png'.format(i))
        image = Image.fromarray((original_image * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)


def generate_images_like_a_batch(model, data_generator, save_dir):
    epsilon = 1e-3
    # Generate random latents and interpolation t-values.
    ln = np.random.normal(size=[model.latents_dim])
    latents_t = np.array([ln for _ in range(model.batch_size)])
    lerp_t = np.random.uniform()

    # Generate latents from the data
    original_data = next(data_generator)
    latents_real = model.encode(original_data)

    latents_e0 = slerp(latents_real[0::2], latents_t[1::2], lerp_t)
    latents_e1 = slerp(latents_real[0::2], latents_t[1::2], lerp_t + epsilon)

    latents_e = np.vstack([latents_e0, latents_e1])

    images = model.decode(latents_e).numpy()


    for i, (original_image, synthetic_image) in enumerate(zip(original_data, images)):
        fig_name = os.path.join(save_dir, 'synthetic_image_{:06d}.png'.format(i))
        image = Image.fromarray((synthetic_image * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)

        fig_name = os.path.join(save_dir, 'original_image_{:06d}.png'.format(i))
        image = Image.fromarray((original_image * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)


def generate_images_randomly(model, save_dir):
    # Generate latents from the data
    images = model.generate_random_images(num_images=32).numpy()

    for i, image in enumerate(images):
        fig_name = os.path.join(save_dir, 'random_synthetic_image_{:06d}.png'.format(i))
        image = Image.fromarray((image * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)
