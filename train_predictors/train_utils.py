from copy import deepcopy

import os
import torch
from torch import Tensor, tensor

from .dataset import SpeechDataset


def get_phones() -> tuple[list[str], int, dict[str, Tensor]]:
    """
    Returns the phone map, pad idx and the ground truth derived from
    `files/aligned-phones_ls-train-clean-100.txt`.
    """
    phone_map = open("/cfs/home/u035986/spkanon/spkanon_eval/private_knnvc/files/phones.sym").read().strip().split("\n")
    pad_idx = len(phone_map)
    phones_true = dict()
    for line in open("/cfs/home/u035986/spkanon/spkanon_eval/private_knnvc/files/duplicated_phoneme_transcript.txt"):
        line = line.strip().split()
        fname, phones = line[0], line[1:]
        phones_true[fname] = tensor(
            [int(p) for p_idx, p in enumerate(phones)]#if p_idx % 2 == 0]
        )
    #import pdb; pdb.set_trace()
    return phone_map, pad_idx, phones_true


def get_split_datasets(
    data_dir: str, phones_true: list[Tensor]
) -> dict[str, SpeechDataset]:
    """
    Splits the given dataset into train, val and test sets.
    """
    dataset = SpeechDataset(data_dir, 16000, ".wav")
    datasets = {split: deepcopy(dataset) for split in ["train", "val", "test"]}
    n_tr = int(len(phones_true) * 0.8)
    n_val = int(len(phones_true) * 0.1)
    datasets["train"].audiofiles = datasets["train"].audiofiles[:n_tr]
    datasets["val"].audiofiles = datasets["val"].audiofiles[n_tr : n_tr + n_val]
    datasets["test"].audiofiles = datasets["test"].audiofiles[n_tr + n_val :]
    return datasets
