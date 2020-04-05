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

"""Provides named, gin configurable ground truth data sets."""

from data.gt_load import mpi3d, dsprites, cars3d, dummy_data, norb, shapes3d
import os


def load(dataset_name, dataset_path):
    """Returns ground truth data set based on name.

    Args:
      dataset_name: String with the name of the dataset.

    Raises:
      ValueError: if an invalid data set name is provided.
    """
    #dataset_path = '..\\gt_datasets'

    SCREAM_PATH = os.path.join(
        dataset_path, "scream", "scream.jpg")

    if dataset_name == "dsprites_full":
        DSP_PATH = os.path.join(dataset_path, 'dsprites', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        return dsprites.DSprites(data_path=DSP_PATH, scream_path=SCREAM_PATH, latents_factor_indices=[1, 2, 3, 4, 5])

    elif dataset_name == "dsprites_noshape":
        DSP_PATH = os.path.join(dataset_path, 'dsprites', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        return dsprites.DSprites(data_path=DSP_PATH, scream_path=SCREAM_PATH, latents_factor_indices=[2, 3, 4, 5])

    elif dataset_name == "color_dsprites":
        DSP_PATH = os.path.join(dataset_path, 'dsprites', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        return dsprites.ColorDSprites(data_path=DSP_PATH, scream_path=SCREAM_PATH, latents_factor_indices=[1, 2, 3, 4, 5])

    elif dataset_name == "noisy_dsprites":
        DSP_PATH = os.path.join(dataset_path, 'dsprites', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        return dsprites.NoisyDSprites(data_path=DSP_PATH, scream_path=SCREAM_PATH, latents_factor_indices=[1, 2, 3, 4, 5])

    elif dataset_name == "scream_dsprites":
        DSP_PATH = os.path.join(dataset_path, 'dsprites', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        return dsprites.ScreamDSprites(data_path=DSP_PATH, scream_path=SCREAM_PATH, latents_factor_indices=[1, 2, 3, 4, 5])

    elif dataset_name == "smallnorb":
        SMALLNORB_TEMPLATE = os.path.join(
            dataset_path,
            "small_norb",
            "smallnorb-{}-{}.mat"
        )

        SMALLNORB_CHUNKS = [
             "5x46789x9x18x6x2x96x96-training",
             "5x01235x9x18x6x2x96x96-testing",
         ]
        return norb.SmallNORB(data_template=SMALLNORB_TEMPLATE, data_chuncks=SMALLNORB_CHUNKS)

    elif dataset_name == "cars3d":
        CARS3D_PATH = os.path.join(dataset_path, "cars")
        return cars3d.Cars3D(data_path=CARS3D_PATH)

    elif dataset_name == "mpi3d_toy":
        return mpi3d.MPI3D(data_path=dataset_path, mode="mpi3d_toy")

    elif dataset_name == "mpi3d_realistic":
        return mpi3d.MPI3D(data_path=dataset_path, mode="mpi3d_realistic")

    elif dataset_name == "mpi3d_real":
        return mpi3d.MPI3D(data_path=dataset_path, mode="mpi3d_real")

    elif dataset_name == "shapes3d":
        S3D_PATH = os.path.join(dataset_path,
                                 '3dshapes',
                                 "look-at-object-room_floor-hueXwall-hueXobj-"
                                 "hueXobj-sizeXobj-shapeXview-azi.npz")

        return shapes3d.Shapes3D(data_path=S3D_PATH)

    elif dataset_name == "dummy_data":
        return dummy_data.DummyData()

    else:
        raise ValueError("Invalid data set name.")
