"""
Create target speakers out of LS' train-clean-100, grouping the features
by their true phone labels.
"""

import os
import sys
import argparse

import torch
from torch import Tensor, tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.cluster import KMeans

try:
    from .train_utils import get_phones
    from .wavlm_loader import load_wavlm
    from .dataset import SpeechDataset
    from .conv_decoder import load_model as load_conv_decoder

except ImportError:
    from train_utils import get_phones
    from wavlm_loader import load_wavlm
    from dataset import SpeechDataset
    from conv_decoder import load_model as load_conv_decoder


def dump_spk_feats(spk: str, feats: list[list[Tensor]], dump_dir: str):
    """
    Dump the features of this speaker. Also, cluster thema and dump the
    cluster centers. Speaker IDs are written to a file.
    """
    for phone_idx in range(len(phone_map)):
        if len(feats[phone_idx]) > 0:
            feats[phone_idx] = torch.stack(feats[phone_idx])
        else:
            feats[phone_idx] = torch.tensor([])

    os.makedirs(os.path.join(dump_dir, "all_feats"), exist_ok=True)

    spk_file = os.path.join(dump_dir, "speakers.txt")
    spk_idx = len(open(spk_file).readlines()) if os.path.exists(spk_file) else 0
    torch.save(feats, os.path.join(dump_dir, "all_feats", f"{spk_idx}.pt"))
    with open(spk_file, "a") as f:
        f.write(f"{spk}\n")

    for n_clusters in [16]:
        clustered_feats = list()
        for phone_idx, phone_feats in enumerate(feats):
            if phone_feats.shape[0] == 0:
                phone_feats = torch.zeros((n_clusters, 1024))
            elif phone_feats.shape[0] < n_clusters:
                phone_feats = torch.cat(
                    [phone_feats] * (n_clusters // phone_feats.shape[0] + 1),
                    dim=0,
                )
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
                phone_feats
            )
            clustered_feats.append(
                torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            )
        clustered_feats = torch.stack(clustered_feats)
        dump_dir_cluster = os.path.join(dump_dir, f"{n_clusters}_clusters")
        os.makedirs(dump_dir_cluster, exist_ok=True)
        torch.save(clustered_feats, os.path.join(dump_dir_cluster, f"{spk_idx}.pt"))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"


    parser = argparse.ArgumentParser(description="Create targets.")
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to the audio data folder"
    )

    parser.add_argument(
        "--opt",
        required=False,           
        default="1",      
        help="Provide own ground truth? 1 - No 2 - Yes"
    )

    try:
        args = parser.parse_args()
        data_dir = args.data_dir
        ground_truth = args.opt
    except SystemExit:
        print("\nWarning: Please add --data_dir <folder_name> to run this script correctly.", file=sys.stderr)
        sys.exit(1)

    print(f"Creating target from data folder: {data_dir}")

    #data_dir = "/ds/audio/LibriSpeech/train-clean-100"
    dump_dir = "./logs/ls-targets-truephones"

    phone_map, pad_idx, phones_true = get_phones()
    wavlm = load_wavlm(device)
    phone_model = load_conv_decoder("./checkpoints/phone_decoder.pt", device)

    dataset = SpeechDataset(data_dir, 16000, ".wav")
    dataset.audiofiles = dataset.audiofiles[:16] # at least 5 min of audio for target
    dl = DataLoader(
        dataset,
        batch_size=8,
        num_workers=10 if device == "cuda" else 0,
        collate_fn=lambda x: [
            pad_sequence([y[0] for y in x], batch_first=True),
            torch.tensor([y[0].shape[0] for y in x]),
            [y[1] for y in x],
        ],
    )


    current_spk = ""
    spk_feats = [list() for _ in range(len(phone_map))]
    for batch_idx, batch in tqdm(enumerate(dl), total=len(dl)):
        audios, lens, fnames = batch
        audios = audios.to(device)
        with torch.no_grad():
            feats = wavlm.extract_features(audios, output_layer=6)[0].to("cpu")
            feats2 = wavlm.extract_features(audios)[0].to(device)

        for fname, utt_feats, utt_feats2 in zip(fnames, feats, feats2):
            fname = os.path.splitext(os.path.basename(fname))[0]

            if ground_truth == "2":
                if fname not in phones_true:
                    #print("Creating phoneme transcript")
                    print(f"{fname} not in phones_true")
                    continue

            else:
                phones_true={}
                with torch.no_grad():
                    utt_feats2 =  utt_feats2.unsqueeze(0)
                    x = phone_model(utt_feats2)
                    
                    x_all = x.argmax(dim=-1).squeeze(0).cpu() 
                    #pred_string = " ".join([str(t) for t in x_all.tolist() if t < len(phone_map)])
                    #print(f"File {fname} -> Phonemes: {pred_string}")
                    print(x_all.tolist())
                    phones_true[fname] = tensor(
                            [int(p) for p_idx, p in enumerate(x_all.tolist())]
                    )
            
            spk = fname.split("_")[0]
            if spk != current_spk:
                if len(spk_feats[0]) > 0:
                    dump_spk_feats(spk, spk_feats, dump_dir)
                current_spk = spk
                spk_feats = [list() for _ in range(len(phone_map))]
            
            for phone, feat in zip(phones_true[fname], utt_feats):
                spk_feats[phone.item()].append(feat)
                
                #pred_string = " ".join([str(phone_map[t]) for t in x_all.tolist() if t < len(phone_map)])
                #print(f"Predicted Phonemes (X) for {fname}:", pred_string)

    if len(spk_feats[0]) > 0:
        dump_spk_feats(spk, spk_feats, dump_dir)
