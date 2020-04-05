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

# Usage of script
# data/download_gt_data.sh -d <Dataset-type> -f <Desired Save directory> -h <help>
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [download_gt_data.sh -d <Dataset-type> -f <Desired Save directory> -h <help>]"
  echo "Options -d [small_norb|cars|dsprites|scream|mpi3d_toy|ALL--default ] -f [cur dir --default]"
  exit 0
fi
DATASET="${DATASET:-ALL}"
FOLDER="${FOLDER:-.}"
while getopts d:f: option
do

case "${option}"
in
d) DATASET=${OPTARG};;
f) FOLDER=${OPTARG};;
?) echo "Unrecognized flag"
esac
done

if [[ ! -d "$FOLDER" ]]; then
  echo "Directory '$FOLDER' does not exists."
  mkdir "$FOLDER"
fi
if [[ ! -d "$FOLDER/.gt_datasets" ]]; then
  #echo "Directory '$FOLDER' does not exists."
  mkdir $FOLDER/.gt_datasets
fi
if [[ "$DATASET" == "small_norb" || "$DATASET" == "ALL" ]]; then
  echo "Downloading small_norb."
  if [[ ! -d "$FOLDER/.gt_datasets/small_norb" ]]; then
    mkdir $FOLDER/.gt_datasets/small_norb
  fi
  if [[ ! -e $FOLDER/.gt_datasets/small_norb/"smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat" ]]; then
    wget -O $FOLDER/.gt_datasets/small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
    gunzip $FOLDER/.gt_datasets/small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
  fi
  if [[ ! -e $FOLDER/.gt_datasets/small_norb/"smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat" ]]; then
    wget -O $FOLDER/.gt_datasets/small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
    gunzip $FOLDER/.gt_datasets/small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
  fi
  if [[ ! -e $FOLDER/.gt_datasets/small_norb/"smallnorb-5x46789x9x18x6x2x96x96-training-info.mat" ]]; then
    wget -O $FOLDER/.gt_datasets/small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
    gunzip $FOLDER/.gt_datasets/small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
  fi
  if [[ ! -e $FOLDER/.gt_datasets/small_norb/"smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat" ]]; then
    wget -O $FOLDER/.gt_datasets/small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
    gunzip $FOLDER/.gt_datasets/small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
  fi
  if [[ ! -e $FOLDER/.gt_datasets/small_norb/"smallnorb-5x46789x9x18x6x2x96x96-testing-cat.mat" ]]; then
    wget -O $FOLDER/.gt_datasets/small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
    gunzip $FOLDER/.gt_datasets/small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
  fi
  if [[ ! -e $FOLDER/.gt_datasets/small_norb/"smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat" ]]; then
    wget -O $FOLDER/.gt_datasets/small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
    gunzip $FOLDER/.gt_datasets/small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
  fi
  echo "Downloading small_norb completed!"
fi

if [[ "$DATASET" == "cars" || "$DATASET" == "ALL" ]]; then
  echo "Downloading cars dataset."
  if [[ ! -d "$FOLDER/.gt_datasets/cars" ]]; then
    mkdir $FOLDER/.gt_datasets/cars
    echo $FOLDER/.gt_datasets/cars/nips2015-analogy-data.tar.gz
    wget -O $FOLDER/.gt_datasets/cars/nips2015-analogy-data.tar.gz http://www.scottreed.info/files/nips2015-analogy-data.tar.gz
    tar xf $FOLDER/.gt_datasets/cars/nips2015-analogy-data.tar.gz
    rm $FOLDER/.gt_datasets/cars/nips2015-analogy-data.tar.gz
    mv data/cars $FOLDER/.gt_datasets/.
    rm -r data
  fi
  echo "Downloading cars completed!"
fi

if [[ "$DATASET" == "dsprites" || "$DATASET" == "ALL" ]]; then
  echo "Downloading dSprites dataset."
  if [[ ! -d "$FOLDER/.gt_datasets/dsprites" ]]; then
    mkdir $FOLDER/.gt_datasets/dsprites
    wget -O $FOLDER/.gt_datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
  fi
  echo "Downloading dSprites completed!"
fi

if [[ "$DATASET" == "scream" || "$DATASET" == "ALL" ]]; then
  echo "Downloading scream picture."
  if [[ ! -d "$FOLDER/.gt_datasets/scream" ]]; then
    mkdir $FOLDER/.gt_datasets/scream
    wget -O $FOLDER/.gt_datasets/scream/scream.jpg https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg
  fi
  echo "Downloading scream completed!"
fi

if [[ "$DATASET" == "mpi3d_toy" || "$DATASET" == "ALL" ]]; then
  echo "Downloading mpi3d_toy dataset."
  if [[ ! -d "$FOLDER/.gt_datasets/mpi3d_toy" ]]; then
    mkdir $FOLDER/.gt_datasets/mpi3d_toy
    wget -O $FOLDER/.gt_datasets/mpi3d_toy/mpi3d_toy.npz https://storage.googleapis.com/disentanglement_dataset/data_npz/sim_toy_64x_ordered_without_heldout_factors.npz
  fi
  echo "Downloading mpi3d_toy completed!"
fi