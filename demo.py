from IPython import display as ipd
import torchaudio
import numpy as np
from demo_model import Converter
import argparse
import sys
import os


rng = np.random.default_rng(seed=42)
device = "cpu"  # runs on CUDA as well

parser = argparse.ArgumentParser(description="Run Anonymization")

parser.add_argument("--clust", required=False, default=[16], help="Provide number of clusters")
parser.add_argument("--weight", required=False, default=[0.3], help="Provide percentage of predicted frame durations")
parser.add_argument("--target_dir", required=False, default="./target_feats",help="Provide path to targets")
parser.add_argument("--path", required=True, help="Provide path to audios or wav.scp")

try:
    args = parser.parse_args()
    cluster_val = args.clust
    weight_val = args.weight
    targets_dir = args.target_dir
    data_dir = args.path
except SystemExit:
    print("\nWarning: Please add --path <folder_name> to run this script.", file=sys.stderr)
    sys.exit(1)


targets = []
for i in os.listdir(targets_dir):
    targets.append(i)

folder_path_source = data_dir
print("Starting inference")
    
for n_clusters in cluster_val:
    for dur_w in weight_val:
        results_dir = f"results_{n_clusters}_{dur_w}"
        os.makedirs(results_dir, exist_ok=True)
        model = Converter(device, dur_w, n_clusters)
        if folder_path_source.endswith("wav.scp"):
            with open(folder_path_source, "r") as f:
                for line in f:
                    utt_name, full_path = line.strip().split(maxsplit=1)
                    if os.path.exists(os.path.join(results_dir,f"{utt_name}.wav")):
                        print(f"{utt_name} already anonymized")
                        continue
                    audio, sr = torchaudio.load(full_path)
                    target_id = rng.choice(targets)
                    try:
                        audio_anon = model.run(audio, target_id) 
                        spk_id = os.path.splitext(target_id)[0]
                        torchaudio.save(os.path.join(results_dir,f"{utt_name}.wav"), audio_anon.unsqueeze(0), 16000)
                    except IndexError as e:
                        print(f"\n[ERROR] IndexError caught on utterance: {utt_name}")
                        print(f"[ERROR] Full path: {full_path}")
                        print(f"[ERROR] Target ID: {target_id}")
                        print(f"[ERROR] Details: {e}\n")
                        continue


        else:
            for i in os.listdir(folder_path_source):  
                if i.endswith(".wav"):
                    full_path = os.path.join(folder_path_source, i)
                    audio, sr = torchaudio.load(full_path)
                    target_id = rng.choice(targets)
                    try:
                        audio_anon = model.run(audio, target_id)
                        spk_id = os.path.splitext(target_id)[0]
                        torchaudio.save(os.path.join(results_dir,f"{i}"), audio_anon.unsqueeze(0), 16000)

                    except IndexError as e:
                        print(f"\n[ERROR] IndexError caught on utterance: {utt_name}")
                        print(f"[ERROR] Full path: {full_path}")
                        print(f"[ERROR] Target ID: {target_id}")
                        print(f"[ERROR] Details: {e}\n")
                        continue
