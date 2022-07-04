# Torch Voice activity detection

The project follows this repository with some minor changes as follows:

1. Some parts of the code were removed
2. The code was refactored at some places because it used many _implicit_ imports, explicit imports and function initialization are better
3. Some of the hyperparameters were changed
4. NOTE: I did not add comments as they should be due to **time limitations**

**One of the reasons I used this solution** the problem requires the knowlodge of quite many hyperparameters and small details which I do not know, since it is a new problem for me.
Therefore, starting from ready solution was a better option in time limited conditions.

### Training

- For training set was used only https://www.openslr.org/resources/12/train-clean-360.tar.gz
- For noise datastes I used thse two datasets:
  1. https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_train.zip?download=1
  2. https://zenodo.org/record/3670167/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.1.zip?download=1
- Training was performed for 100 epochs
- To obtain true labels WebRTC was used on clean _train-clean-360_
- Training was done with the following settings and AdamW + ReduceLROnPlateau was used as LR Scheduler:

```yaml
batch_size: 32
lr: 1e-4
weight_decay: 1e-3
```

To check other training settings I refer to ./config/config.yaml

## Results

For validation set I used 0.2 noisy portion of train-clean-360.
Data augmentation for validation set was performed randmoly during validation, I am aware that it is not correct but It provides with some reliable estimates.

| EER  | FPR with FNR = 1% | FNR with FPR = 1% | F1   |
| ---- | ----------------- | ----------------- | ---- |
| 0.45 | 0.96              | 0.92%             | 0.96 |

Results on required dataset can be found in './results'

I consider metrics used in the original implementation to be reasonable. EER, FPR with 1% FNR and FNR with 1% FPR was calculated during validation. More details src/utils/metrics.py

## Lattency

Lattency was measured per file and it includes data laoding time. The current lattency per file is ~ 30ms.

## Training a model

- download all the datasets using **get_data.sh**
- preprocess LibriSpeech dataset

```
python scripts/preprocessing_dataset.py -D path_to_librispeech_dataset
```

- change paths to datasets in ./config/config.yaml
- install all the dependencies from requirements.txt
- run train.py -g GPU (Where gpu is your favourite gpu)

## Inference

To obtain resilts on your own dataset

- add a path to your dataset folder (\*.wav files) in ./config/config.yaml at data.test_files_path
- run the following command:

```
python inference.py -g YOUR_GPU -chk  path_to_stored_model -dir outptut_dir
```

### Possible improvements

- USE distilation with SILERO
- Different Model design
