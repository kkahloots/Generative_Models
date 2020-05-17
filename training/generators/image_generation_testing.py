
import numpy as np
import tensorflow as tf
from evaluation.generativity_metrics.shared_api import slerp
from PIL import Image

from collections import defaultdict
from tqdm import tqdm
import os

def reconstruct_from_a_batch(model, data_generator, save_dir):
    # Generate latents from the data
    original_data = next(data_generator)
    if isinstance(original_data, tuple):
        original_data = original_data[0]
    images = model.reconstruct(original_data).numpy()

    for i, (original_image, reconstructed_image) in tqdm(enumerate(zip(original_data, images)), position=0):
        fig_name = os.path.join(save_dir, 'reconstructed_image_{:06d}.png'.format(i))
        image = Image.fromarray((reconstructed_image * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)

        fig_name = os.path.join(save_dir, 'original_image_{:06d}.png'.format(i))
        image = Image.fromarray((original_image.numpy() * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)


def predict_from_a_batch(model, data_generator, save_dir):
    # Generate from the data
    original_data = next(data_generator)
    data_xt0 = original_data[0]
    data_xt1 = original_data[1]
    data_xt0 = tf.reshape(data_xt0, [model.batch_size, -1,] + model.get_inputs_shape())
    data_xt1 = tf.reshape(data_xt1, [model.batch_size, -1,] + model.get_inputs_shape())

    data_xt0_pred = model.predict(original_data[0])
    data_xt0_pred = tf.reshape(data_xt0_pred, [model.batch_size, -1,] + model.get_inputs_shape())

    for i, (xt0, xt1, xt0_pred) in tqdm(enumerate(zip(data_xt0, data_xt1, data_xt0_pred)), position=0):
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
    lerp_t = np.random.uniform(size=1)[0]
    #lerp_t = np.array([lerp_t for _ in range(latents_real.shape[0])])
    latents_e = slerp(lerp_t, latents_real, latents_t)
    images = model.decode(latents_e).numpy()

    for i, (original_image, synthetic_image) in tqdm(enumerate(zip(original_data, images)), position=0):
        fig_name = os.path.join(save_dir, 'synthetic_image_{:06d}.png'.format(i))
        image = Image.fromarray((synthetic_image * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)

        fig_name = os.path.join(save_dir, 'original_image_{:06d}.png'.format(i))
        image = Image.fromarray((original_image.numpy() * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)


def generate_images_randomly(model, save_dir):
    # Generate latents from the data
    images = model.generate_random_images(num_images=32).numpy()

    for i, image in tqdm(enumerate(images), position=0):
        fig_name = os.path.join(save_dir, 'random_synthetic_image_{:06d}.png'.format(i))
        image = Image.fromarray((image * 255).astype(np.uint8), mode='RGB')
        image.save(fig_name)



def interpolate_a_batch(model, data_generator, save_dir, delay=10):
    # Generate latents from the data
    original_data = next(data_generator)
    if isinstance(original_data, tuple):
        original_data = original_data[0]
    images = []
    last_ix = 0
    for ix in tqdm(range(0, len(original_data) // 2 * 2 - 2, 2), position=0):
        images += interpolate(model, original_data[last_ix:ix + 1], original_data[ix + 1:ix + 2])
        last_ix = ix + 1
    images += interpolate(model, original_data[ix + 1:ix + 2], original_data[0:1])

    images_flat = []
    i = 0
    for image_raw in images:
        for image in image_raw[:-1]:
            # fig_name = os.path.join(save_dir, 'original_image_{:06d}.png'.format(i))
            image = Image.fromarray((image * 255).astype(np.uint8), mode='RGB')
            # image.save(fig_name)
            images_flat += [image]
            i += 1

    movie_name = os.path.join(save_dir, model.name + '_interpolate.gif')
    images_flat[0].save(movie_name, save_all=True, append_images=images_flat[1:], duration=len(images_flat) * delay,
                        loop=0xffff)

def interpolate(model, input1, input2):

    z1 = model.encode(input1).numpy()
    z2 = model.encode(input2).numpy()

    decodes = defaultdict(list)
    for idx, ratio in tqdm(enumerate(np.linspace(0, 1, 10)), position=0):
        decode = dict()
        z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
        z_decode = model.decode(z).numpy()

        for i in range(z_decode.shape[0]):
            decode[i] = [z_decode[i]]

        for i in range(z_decode.shape[0]):
            decodes[i] = decodes[i] + decode[i]

    imgs = []

    for idx in decodes:
        l = []
        l += [input1[idx:idx + 1][0]]
        l += decodes[idx]
        l += [input2[idx:idx + 1][0]]

        imgs.append(l)

    return imgs