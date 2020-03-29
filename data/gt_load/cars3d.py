# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cars3D data set."""

import os
from data.gt_load import gt_data, util
import numpy as np
import PIL
import scipy.io as sio
from six.moves import range
from sklearn.utils.extmath import cartesian
import tensorflow as tf

#dataset_path = "..//gt_datasets"
#CARS3D_PATH = os.path.join(dataset_path, "cars")

class Cars3D(gt_data.GroundTruthData):
    """Cars3D data set.

    The data set was first used in the paper "Deep Visual Analogy-Making"
    (https://papers.nips.cc/paper/5845-deep-visual-analogy-making) and can be
    downloaded from http://www.scottreed.info/. The images are rescaled to 64x64.

    The ground-truth factors of variation are:
    0 - elevation (4 different values)
    1 - azimuth (24 different values)
    2 - object type (183 different values)
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.factor_sizes = [4, 24, 183]
        features = cartesian([np.array(list(range(i))) for i in self.factor_sizes])
        self.latents_factor_indices = [0, 1, 2]
        self.num_total_factors = features.shape[1]
        self.index = util.StateSpaceAtomIndex(self.factor_sizes, features)
        self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latents_factor_indices)
        self.data_shape = [64, 64, 3]
        self.images = self._load_data()

    @property
    def num_factors(self):
        return self.state_space.num_latents_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latents_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = self.index.features_to_index(all_factors)
        return self.images[indices].astype(np.float32)

    def _load_data(self):
        dataset = np.zeros((24 * 4 * 183, 64, 64, 3))
        all_files = [x for x in tf.io.gfile.listdir(self.data_path) if ".mat" in x]
        for i, filename in enumerate(all_files):
            data_mesh = _load_mesh(filename)
            factor1 = np.array(list(range(4)))
            factor2 = np.array(list(range(24)))
            all_factors = np.transpose([
                np.tile(factor1, len(factor2)),
                np.repeat(factor2, len(factor1)),
                np.tile(i,
                        len(factor1) * len(factor2))
            ])
            indexes = self.index.features_to_index(all_factors)
            dataset[indexes] = data_mesh
        return dataset


def _load_mesh(filename, data_path):
    """Parses a single source file and rescales contained images."""
    with tf.io.gfile.GFile(os.path.join(data_path, filename), "rb") as f:
        mesh = np.einsum("abcde->deabc", sio.loadmat(f)["im"])

    flattened_mesh = mesh.reshape((-1,) + mesh.shape[2:])
    rescaled_mesh = np.zeros((flattened_mesh.shape[0], 64, 64, 3))

    for i in range(flattened_mesh.shape[0]):
        pic = PIL.Image.fromarray(flattened_mesh[i, :, :, :])
        pic.thumbnail((64, 64, 3), PIL.Image.ANTIALIAS)
        rescaled_mesh[i, :, :, :] = np.array(pic)
    return rescaled_mesh * 1. / 255
