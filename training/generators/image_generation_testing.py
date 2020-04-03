from PIL import Image
import os
import numpy as np
import tensorflow as tf
from evaluation.generativity_metrics.shared_api import slerp

def reconstruct_from_a_batch(model, data_generator, save_dir):
    # Generate latents from the data
    original_data = next(data_generator)
    if isinstance(original_data, tuple):
        original_data = original_data[0]
    images = model.reconstruct(original_data).numpy()

    for i, (original_image, reconstructed_image) in enumerate(zip(original_data, images)):
        fig_name = os.path.join(save_dir, 'reconstructed_image_{:06d}.png'.format(i))
        image = Image.fromarray((reconstructed_image * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)

        fig_name = os.path.join(save_dir, 'original_image_{:06d}.png'.format(i))
        image = Image.fromarray((original_image * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)


def predict_from_a_batch(model, data_generator, save_dir):
    # Generate from the data
    original_data = next(data_generator)
    data_xt0 = original_data[0]
    data_xt1 = original_data[1]
    data_xt0 = tf.reshape(data_xt0, [model.batch_size, -1,] + model.get_input_shape())
    data_xt1 = tf.reshape(data_xt1, [model.batch_size, -1,] + model.get_input_shape())

    data_xt0_pred = model.predict(original_data[0])
    data_xt0_pred = tf.reshape(data_xt0_pred, [model.batch_size, -1,] + model.get_input_shape())

    for i, (xt0, xt1, xt0_pred) in enumerate(zip(data_xt0, data_xt1, data_xt0_pred)):
        xt0 = tf.concat([x for x in xt0], axis=1).numpy()
        xt1 = tf.concat([x for x in xt1], axis=1).numpy()

        xt0_pred = tf.concat([x for x in xt0_pred], axis=1).numpy()

        fig_name = os.path.join(save_dir, 'input_image_{:06d}.png'.format(i))
        image = Image.fromarray((xt0 * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)

        fig_name = os.path.join(save_dir, 'output_image_{:06d}.png'.format(i))
        image = Image.fromarray((xt1 * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)

        fig_name = os.path.join(save_dir, 'predicted_image_{:06d}.png'.format(i))
        image = Image.fromarray((xt0_pred * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)

def generate_images_like_a_batch(model, data_generator, save_dir):
    epsilon = 1e-3

    # Generate latents from the data
    original_data = next(data_generator)
    if isinstance(original_data, tuple):
        original_data = original_data[0]
    latents_real = model.encode(original_data)

    # Generate random latents and interpolation t-values.
    ln = np.random.normal(size=[latents_real.shape[1]])
    latents_t = np.array([ln for _ in range(latents_real.shape[0])])
    lerp_t = np.random.uniform()

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
