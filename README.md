# Voice activity Detection

source: https://github.com/ilyailyash/Torch-Voice-activity-detection

The project follows the repository above with some changes as follows:

1. Some parts of the code were removed
2. The code was refactored at some places because it used many **_implicit_** imports, explicit imports and function initialization are better
3. Some of the hyperparameters were changed
4. NOTE: I did not add comments as they should be due to **time limitation**

**One of the reasons I used this solution** the problem requires the knowlodge of quite many hyperparameters and small details which I do not know, since it is a new problem for me.
Therefore, starting from ready solution was a better option in time limited conditions.

### Training

- For training set was used only https://www.openslr.org/resources/12/train-clean-360.tar.gz
- For noise datastes I used thse two datasets:
  1. https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_train.zip?download=1
  2. https://zenodo.org/record/3670167/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.1.zip?download=1
- Training was performed for 100 epochs
- To obtain true labels WebRTC was used on clean _train-clean-360_
- Training was done with the following settings and AdamW + ReduceLROnPlateau as LR Scheduler:

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

![metrics](metrics.png)

From the plots it can be seen that the training procedure could be improved.
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
- run train.py -g YOUR_GPU (Where YOUR_GPU is your favourite gpu)

## Inference

To obtain results on your own dataset

- Add a path to your dataset folder (\*.wav files) in ./config/config.yaml at data.test_files_path
- Run the following command:

```
python inference.py -g YOUR_GPU -chk  path_to_stored_model -dir outptut_dir
```

### Possible improvements & Model choice

There are several ways to improve model perfromance:
- Change model desing. But need to keep the model fast enough.
Current model was taken from here: https://github.com/liyaguang/DCRNN.
- Use distilation e.g. with SILERO or other models, enmsemble of models as teachers 
- Different augmentaions, datasets and training settings

One of possible ways to augment time series data is the following:
```python
def random_window_transform(x, seq_len=0, window=5):
    device = x.device
    coefs = [random.random() for _ in range(window)]
    weight = torch.tensor([[coefs]])
    x = torch.nn.functional.conv1d(
        x, weight, bias=None, stride=1, padding=window // 2
    )
    return x
```

It creates a random convolution with fixed window length. Conv weights can be limited to avoid strong modifications. Unfortunately, I have not tested it with the current problem.


## Models review:
There are several types of models for VAD problem, some of them DNN based and some are not and rely on classical solutions (WebRTC - GMM). One of the main challenges for VAD is noise and signal strength, here DNN based solutions excel the most. Usually, DNN based solutions are combination of CNN and RNN layers or only one from the both.  

Atention and self attention was recently explored for VAD problem e.g.: 
https://arxiv.org/pdf/2203.02944.pdf. Surprisingly, models with self attention demonstrate reasonable computational efficiency.

Another interesting direction of work is Multilingual VAD systemes (https://arxiv.org/abs/2010.12277) and unsuperwised approaches. 

Since, VAD models require high computational efficiency it is reasonable to apply NAS (Neural architechture search for this problem) with computaional constrains. Authors in https://arxiv.org/pdf/2201.09032.pdf explored NAS for VAD but unfortunately without computational constrains.