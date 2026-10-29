# Voice Anonymization Model for European Portuguese

Anonymize speech by replacing each phoneme-level frame with features from a chosen target speaker.

---

## Setup

### 1. Clone the repo and checkpoints

Clone this repository, then download the checkpoints from the **Releases** page.
After unzipping, the folder must be named `checkpoints`.

```
repo/
├── checkpoints/
├── train_predictors/
├── demo.py
└── ...
```

### 2. Create a virtual environment

```bash
python -m venv anonvenv
source anonvenv/bin/activate        # Windows: anonvenv\Scripts\activate
pip install -r requirements.txt
```

---
For the following procedures make sure all your audio files have a 16kHz sampling rate.

## Creating Targets

You have two options for building target speaker features:

| Option | You provide                        |
|--------|------------------------------------|
| **1**  | Audio + aligned phoneme data       |
| **2**  | Audio only                         |

### Option 1: Audio + aligned phonemes

Place your aligned data in the `files/` folder as `duplicated_phoneme_transcript.pt`.

Each line has the format:

```
<filename> <duplicated_phoneme_sequence>
```

The phoneme sequence has **one phoneme per 20 ms frame**, separated by spaces.

Example:

```
utt_0001 sil sil sil l l l a a a a sil sil
```

### Option 2: Audio only

This is the default option, no need to specify 

Then, from the repo directory, run:

```bash
python train_predictors/targets_truephones.py --data_dir <folder_name>
```

Targets are saved to:

```
logs/ls-targets-truephones/all_feats
```

### Select your target speakers

Copy the target speakers you want to use for anonymization into `./target_feats` in the main directory.

You're now ready to use the model. 

---

##  Running the Model

Provide either a `wav.scp` file or a folder of audio files:

```bash
python demo.py --path <wav.scp or folder_path>
```

Output audio is written to a folder named:

```
results_<nclust>_<weight>
```

### Optional arguments

You can also set the number of clusters, the weights, and the target directory from the command line. See all options with:

```bash
python demo.py --help
```

---

## 📁 Project Layout

```
.
├── checkpoints/          # downloaded model checkpoints
├── files/                # aligned phoneme data (Option 1)
├── target_feats/         # chosen target speakers
├── logs/                 # generated targets
├── train_predictors/     # target-building scripts
├── demo.py               # anonymization entry point
└── requirements.txt
```

## Ethical Disclaimer 
We release this technology to support future research on voice anonymization, and we expect that most of its applications will be beneficial.

##  Acknowledgements

This project builds on **Private kNN-VC** and **kNN-VC**. The phone predictor, WavLM feature pipeline, and kNN-based conversion follow their approach. Please cite their work if you use this code:

- Official repository: [carlosfranzreb/private_knnvc](https://github.com/carlosfranzreb/private_knnvc)

```bibtex
@inproceedings{franzreb2025private,
  title     = {Private kNN-VC: Interpretable Anonymization of Converted Speech},
  author    = {Franzreb, Carlos and Das, Arnab and Polzehl, Tim and M{\"o}ller, Sebastian},
  booktitle = {Interspeech 2025},
  pages     = {3224--3228},
  year      = {2025},
  publisher = {ISCA}
}

@inproceedings{baas2023knnvc,
  title     = {Voice Conversion With Just Nearest Neighbors},
  author    = {Baas, Matthew and van Niekerk, Benjamin and Kamper, Herman},
  booktitle = {Interspeech 2023},
  year      = {2023}
}
```
This work was developed in collaboration with INESC-ID as a Master Thesis project. 
A special thank you to Isabel Trancoso, Francisco Teixeira for advising the whole process
