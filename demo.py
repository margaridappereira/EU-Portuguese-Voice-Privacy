from IPython import display as ipd
import torchaudio
import numpy as np
from demo_model import Converter
import sys
import os

rng = np.random.default_rng(seed=42)
device = "cpu"  # runs on CUDA as well
cluster_val = [16] #[0, 8, 16, 32, 64, 128, 256]
weight_val = [0.3] #[0, 0.3, 0.7, 1]
targets = ["0.pt", "1.pt", "2.pt", "3.pt", "4.pt", "5.pt", "6.pt", "7.pt", "8.pt", "9.pt", "10.pt", "11.pt"]
dataset_asr = [ 'portugal_2', 'portugal_3', 'brasil_2', 'brasil_3']
#dataset_asv = ['angola', 'brasil', 'guine', 'mocambique', 'portugal']
#dataset_asr =['7925','3050']

#for data_asr in dataset_asr:
for data_asr in dataset_asr:
    #folder_path_source = rf'/cfs/home/u035986/multi_speaker/Voice-Privacy-Challenge-2024/adapted_utils/test_folders/varients/varients_{data_asv}/wav.scp'
    #folder_path_source = rf'/cfs/home/u035986/multi_speaker/Voice-Privacy-Challenge-2024/data/{data_asr}/wav.scp'
    #folder_path_source = rf'/cfs/home/u035986/multi_speaker/Voice-Privacy-Challenge-2024/data_wer/{data_asr}_test_trials/wav.scp'  
    folder_path_source = rf'/cfs/home/u035986/pt-audio_wav_16k/{data_asr}/wav.scp'
    #folder_path_source = rf'/cfs/home/u035986/pt-audio_wav_16k/pt-mls/data_pt-br_f/{data_asr}/wav.scp'
    print("Starting inference for dataset", dataset_asr)
    #print("Starting inference for dataset", data_asv)  
#"/cfs/projects/funded/accelerat/ASR/data/parlamento/alignments/FalaAR_segments_test"#sys.argv[1]
    #os.makedirs(f"exp_clustweight_{data_asr}", exist_ok=True) 
    os.makedirs(f"exp_clustweight_{data_asr}", exist_ok=True) 
    for n_clusters in cluster_val:
        for dur_w in weight_val:
            model = Converter(device, dur_w, n_clusters)
        with open(folder_path_source, "r") as f:
            for line in f:
        #for i in os.listdir(folder_path_source):
                utt_name, full_path = line.strip().split(maxsplit=1)
                if os.path.exists(f"exp_clustweight_{data_asr}/{utt_name}.wav"):
                    print(f"{utt_name} already anonymized")
                    continue
                audio, sr = torchaudio.load(full_path)
            #if i.endswith(".wav"):
            #    full_path = os.path.join(folder_path_source, i)
            #    audio, sr = torchaudio.load(full_path)

                target_id = rng.choice(targets)
                try:
                    audio_anon = model.run(audio, target_id)
                    spk_id = os.path.splitext(target_id)[0]
                #split_parts = i.split("_")
                #utt_name = f"{split_parts[0]}_{split_parts[1]}_{split_parts[2]}_{split_parts[3]}_{split_parts[4]}"
                    torchaudio.save(f"exp_clustweight_{data_asr}/{utt_name}.wav", audio_anon.unsqueeze(0), 16000)

                except IndexError as e:
                    print(f"\n[ERROR] IndexError caught on utterance: {utt_name}")
                    print(f"[ERROR] Full path: {full_path}")
                    print(f"[ERROR] Target ID: {target_id}")
                    print(f"[ERROR] Details: {e}\n")
                    continue
