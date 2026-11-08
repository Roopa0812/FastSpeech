# FastSpeech 2 Text-to-Speech for English and Telugu

Two FastSpeech 2 text-to-speech systems trained from scratch on about five hours of selected speech per language:

- **English (Indian-accented):** Svarah corpus, 60 approximated speakers
- **Telugu:** IndicTTS Telugu corpus, 2 speakers (TEF and TEM)

The implementation is based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), with patches for the selected datasets and a new 47-phone Telugu front-end. A pretrained universal HiFi-GAN vocoder is used unchanged.

## Summary

| Measure | English | Telugu |
|---|---|---|
| Corpus | Svarah | IndicTTS Telugu |
| Speakers used | 60 | 2 (TEF, TEM) |
| Aligned speech (train + validation) | 5.66 h | 4.23 h |
| Acoustic-model parameters | 35.43 M | 35.35 M |
| Final training step | 15,000 | 22,000 |
| Best validation total loss | 2.639 (step 9,000) | 3.768 (step 1,000) |

Final checkpoints used for synthesis: English step 15,000 (speaker S000), Telugu step 22,000 (speaker TEM).

## Main changes to the reference implementation

- Separate utterance-level train/validation lists; pitch and energy statistics computed from training data only
- English: pause-aware alignment (gaps of at least 30 ms become an `sp` token)
- Telugu: rule-based script-to-phone converter (47 phones) and a Telugu number reader for numerals, ordinals and times
- Corpus-specific Montreal Forced Aligner models trained for each language
- Compatibility fixes for current PyTorch and librosa; CPU-friendly feature extraction
- New `inference.py` that saves audio, duration/pitch/energy plots, mel plots and per-phoneme CSV files

## Repository structure

```
FastSpeech
├── README.md
├── requirements.txt
├── .gitignore
├── src/            # inference.py
├── configs/
│   ├── config_Svarah/    # English configs
│   └── config_Telugu/    # Telugu configs
├── dictionaries/   # pronunciation dictionaries and word/dict ID files
├── metadata/       # corpus audits, train/validation splits
├── notebooks/
└── patches/        # diffs against ming024/FastSpeech2
```

| Folder | Contents |
|---|---|
| `src/` | `inference.py` |
| `configs/` | `config_Svarah/`, `config_Telugu/` |
| `dictionaries/` | `dict.txt`, `dict_ids.txt`, `word_ids.tsv` |
| `metadata/` | `svarah_audit.csv`, `telugu_audit.csv`, `svarah_split.csv`, `telugu_split.csv`, `split_val_utts.txt` |
| `patches/` | `repo_patches.diff`, `telugu_symbols.diff` |

## Setup

This repo holds only the changes, not a full copy of the original code. To reproduce:

1. Clone the reference implementation:
   ```bash
   git clone https://github.com/ming024/FastSpeech2.git
   cd FastSpeech2
   ```
2. Apply the patches from this repo:
   ```bash
   git apply ../FastSpeech/patches/repo_patches.diff
   git apply ../FastSpeech/patches/telugu_symbols.diff
   ```
3. Copy `src/inference.py` into the cloned folder and the config folders from `configs/` into its `config/` folder.
4. Install dependencies: `pip install -r requirements.txt`
5. Install Montreal Forced Aligner 3.4.3 separately (conda-forge, via micromamba). It is not a pip dependency.

## Main commands

```bash
# 1. Train the forced aligner for the selected language
mfa train --clean -j 2 --output_directory preprocessed_data/<Dataset>/TextGrid \
  raw_data/<Dataset> <dictionary.txt> <dataset>_acoustic.zip

# 2. Extract mel, pitch, energy and duration features
python preprocess.py config/<Dataset>/preprocess.yaml

# 3. Train the model (add --restore_step N to resume)
python train.py -p config/<Dataset>/preprocess.yaml \
  -m config/<Dataset>/model.yaml \
  -t config/<Dataset>/train.yaml

# 4. English inference
python inference.py --lang english --dataset Svarah --restore_step 15000 \
  --speaker S000 --out Results/English

# 5. Telugu inference
python inference.py --lang telugu --dataset Telugu --restore_step 22000 \
  --speaker TEM --sentences test_sentences_telugu.json --out Results/Telugu
```

`<Dataset>` is `Svarah` for English and `Telugu` for the Telugu corpus.

## Environment

Experiments ran on Google Colab (Python 3.13; Python 3.11 in the MFA environment). Forced alignment, feature extraction and inference ran on CPU; training used a Tesla T4 GPU.

## Limitations

- Small training sets compared with the original FastSpeech 2 setup
- Telugu uses only two speakers, and TEF recordings are noisier than TEM
- Informal listening assessment by one listener; no MOS/CMOS study
- Telugu does not yet use the pause-aware feature extraction applied to English
- Telugu number reader and rare phone sequences need validation by native speakers

## References

1. Y. Ren et al., "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech," ICLR 2021.
2. J. Kong, J. Kim and J. Bae, "HiFi-GAN," NeurIPS 2020.
3. M. McAuliffe et al., "Montreal Forced Aligner," Interspeech 2017.
4. M. Morise et al., "WORLD vocoder," IEICE Trans. Inf. Syst., 2016.
5. T. Javed et al., "Svarah: Evaluating English ASR Systems on Indian Accents," Interspeech 2023.
6. A. Baby et al., "Resources for Indian Languages," 2016.
7. ming024, FastSpeech2 (PyTorch), github.com/ming024/FastSpeech2.
8. K. Park and J. Kim, g2p_en, github.com/Kyubyong/g2p.
