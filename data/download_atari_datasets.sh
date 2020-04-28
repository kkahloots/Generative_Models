#!/bin/bash

# Usage of script
# data/download_gt_data.sh -d <Dataset-type> -f <Desired Save directory> -h <help>
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [download_atari_dataset.sh -d <Dataset-type> -f <Desired Save directory> -h <help>]"
  echo "Options -d [mspacman|qbert|pinball|revenge|spaceinvaders|ALL--default ] -f [cur dir --default]"
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
if [[ ! -d "$FOLDER/.atari_datasets" ]]; then
  #echo "Directory '$FOLDER' does not exists."
  mkdir $FOLDER/.atari_datasets
fi
if [[ "$DATASET" == "mspacman" || "$DATASET" == "ALL" ]]; then
  echo "Downloading mspacman."
  if [[ ! -d "$FOLDER/.atari_datasets/mspacman" ]]; then
    mkdir $FOLDER/.atari_datasets/mspacman
  fi
  if [[ ! -e $FOLDER/.atari_datasets/mspacman/"mspacman.txt" ]]; then
    wget https://omnomnom.vision.rwth-aachen.de/data/atari_v1_release/mspacman.tar.gz -P $FOLDER/.atari_datasets/mspacman/ --no-check-certificate
    tar -xf $FOLDER/.atari_datasets/mspacman/mspacman.tar.gz -C $FOLDER/.atari_datasets/mspacman/
    rm -rf $FOLDER/.atari_datasets/mspacman/mspacman.tar.gz
    echo "insert text here" > $FOLDER/.atari_datasets/mspacman/mspacman.txt
    echo "Images can be found at $FOLDER/.atari_datasets/mspacman/atari_v1/screens/mspacman"
  fi
  echo "Downloading mspacman completed!"
fi

if [[ "$DATASET" == "qbert" || "$DATASET" == "ALL" ]]; then
  echo "Downloading qbert."
  if [[ ! -d "$FOLDER/.atari_datasets/qbert" ]]; then
    mkdir $FOLDER/.atari_datasets/qbert
  fi
  if [[ ! -e $FOLDER/.atari_datasets/qbert/"qbert.txt" ]]; then
    wget https://omnomnom.vision.rwth-aachen.de/data/atari_v1_release/qbert.tar.gz -P $FOLDER/.atari_datasets/qbert/ --no-check-certificate
    tar -xf $FOLDER/.atari_datasets/qbert/qbert.tar.gz -C $FOLDER/.atari_datasets/qbert/
    rm -rf $FOLDER/.atari_datasets/qbert/qbert.tar.gz
    echo "insert text here" > $FOLDER/.atari_datasets/qbert/qbert.txt
    echo "Images can be found at $FOLDER/.atari_datasets/qbert/atari_v1/screens/qbert"
  fi
  echo "Downloading qbert completed!"
fi

if [[ "$DATASET" == "pinball" || "$DATASET" == "ALL" ]]; then
  echo "Downloading pinball."
  if [[ ! -d "$FOLDER/.atari_datasets/pinball" ]]; then
    mkdir $FOLDER/.atari_datasets/pinball
  fi
  if [[ ! -e $FOLDER/.atari_datasets/pinball/"pinball.txt" ]]; then
    wget https://omnomnom.vision.rwth-aachen.de/data/atari_v1_release/pinball.tar.gz -P $FOLDER/.atari_datasets/pinball/ --no-check-certificate
    tar -xf $FOLDER/.atari_datasets/pinball/pinball.tar.gz -C $FOLDER/.atari_datasets/pinball/
    rm -rf $FOLDER/.atari_datasets/pinball/pinball.tar.gz
    echo "insert text here" > $FOLDER/.atari_datasets/pinball/pinball.txt
    echo "Images can be found at $FOLDER/.atari_datasets/pinball/atari_v1/screens/pinball"
  fi
  echo "Downloading pinball completed!"
fi

if [[ "$DATASET" == "revenge" || "$DATASET" == "ALL" ]]; then
  echo "Downloading revenge."
  if [[ ! -d "$FOLDER/.atari_datasets/revenge" ]]; then
    mkdir $FOLDER/.atari_datasets/revenge
  fi
  if [[ ! -e $FOLDER/.atari_datasets/revenge/"revenge.txt" ]]; then
    wget https://omnomnom.vision.rwth-aachen.de/data/atari_v1_release/revenge.tar.gz -P $FOLDER/.atari_datasets/revenge/ --no-check-certificate
    tar -xf $FOLDER/.atari_datasets/revenge/revenge.tar.gz -C $FOLDER/.atari_datasets/revenge/
    rm -rf $FOLDER/.atari_datasets/revenge/revenge.tar.gz
    echo "insert text here" > $FOLDER/.atari_datasets/revenge/revenge.txt
    echo "Images can be found at $FOLDER/.atari_datasets/revenge/atari_v1/screens/revenge"
  fi
  echo "Downloading revenge completed!"
fi

if [[ "$DATASET" == "spaceinvaders" || "$DATASET" == "ALL" ]]; then
  echo "Downloading spaceinvaders."
  if [[ ! -d "$FOLDER/.atari_datasets/spaceinvaders" ]]; then
    mkdir $FOLDER/.atari_datasets/spaceinvaders
  fi
  if [[ ! -e $FOLDER/.atari_datasets/spaceinvaders/"spaceinvaders.txt" ]]; then
    wget https://omnomnom.vision.rwth-aachen.de/data/atari_v1_release/spaceinvaders.tar.gz -P $FOLDER/.atari_datasets/spaceinvaders/ --no-check-certificate
    tar -xf $FOLDER/.atari_datasets/spaceinvaders/spaceinvaders.tar.gz -C $FOLDER/.atari_datasets/spaceinvaders/
    rm -rf $FOLDER/.atari_datasets/spaceinvaders/spaceinvaders.tar.gz
    echo "insert text here" > $FOLDER/.atari_datasets/spaceinvaders/spaceinvaders.txt
    echo "Images can be found at $FOLDER/.atari_datasets/spaceinvaders/atari_v1/screens/spaceinvaders"
  fi
  echo "Downloading spaceinvaders completed!"
fi