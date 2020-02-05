#!/bin/bash
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

mkdir .gt_datasets

echo "Downloading small_norb."
cd .gt_datasets
mkdir small_norb
cd small_norb
wget -O smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
gunzip smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz

wget -O smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
gunzip smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz

wget -O smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
gunzip smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz

wget -O smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
gunzip smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz

wget -O smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
gunzip smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz

wget -O smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
gunzip smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz

echo "Downloading small_norb completed!"
cd ..
echo "Downloading cars dataset."

mkdir cars
cd cars
wget -O nips2015-analogy-data.tar.gz http://www.scottreed.info/files/nips2015-analogy-data.tar.gz
tar -xf nips2015-analogy-data.tar.gz
rm nips2015-analogy-data.tar.gz
echo "Downloading cars completed!"

cd ..
echo "Downloading dSprites dataset."
mkdir dsprites
cd dsprites
wget -O dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz

echo "Downloading dSprites completed!"

cd ..
echo "Downloading scream picture."
mkdir scream
cd scream
wget -O scream.jpg https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg
echo "Downloading scream completed!"

echo "Downloading mpi3d_toy dataset."
cd ..
mkdir mpi3d_toy
cd mpi3d_toy
wget -O mpi3d_toy.npz https://storage.googleapis.com/disentanglement_dataset/data_npz/sim_toy_64x_ordered_without_heldout_factors.npz

echo "Downloading mpi3d_toy completed!"
cd ..
cd ..