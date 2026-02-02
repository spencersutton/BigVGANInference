# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import os

import numpy as np
import torch

MAX_WAV_VALUE = 32767.0  # NOTE: 32768.0 -1 to prevent int16 overflow (results in popping sound in corner cases)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


def get_dataset_filelist(a):
    training_files = []
    validation_files = []
    list_unseen_validation_files = []

    with open(a.input_training_file, encoding="utf-8") as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav") for x in fi.read().split("\n") if len(x) > 0]
        print(f"first training file: {training_files[0]}")

    with open(a.input_validation_file, encoding="utf-8") as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav") for x in fi.read().split("\n") if len(x) > 0]
        print(f"first validation file: {validation_files[0]}")

    for i in range(len(a.list_input_unseen_validation_file)):
        with open(a.list_input_unseen_validation_file[i], encoding="utf-8") as fi:
            unseen_validation_files = [os.path.join(a.list_input_unseen_wavs_dir[i], x.split("|")[0] + ".wav") for x in fi.read().split("\n") if len(x) > 0]
            print(f"first unseen {i}th validation fileset: {unseen_validation_files[0]}")
            list_unseen_validation_files.append(unseen_validation_files)

    return training_files, validation_files, list_unseen_validation_files
