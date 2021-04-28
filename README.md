# Code for BirdCLEF 2021 Kaggle Competition

<https://www.kaggle.com/c/birdclef-2021>

## Plan

### Dataset Creation

- read `train_metadata.csv`
- select only labels with good rating
- slice clips into fragments with `stride` and `length` or cut based on detection model
- randomly drop samples classes with too much data
- "upsample" classes with too little data
- convert coordinates into orthogonal basis,  bin them to a coarse grid (10x10?)
- add date coarsened up to season (month?, 1/8 of y?)

---

- read `train_soundscape_labels.csv`
- add date coarsened date
- add coarsened coordinates
- assume rating is '5' (?)
- assume all labels are primary (?)

---

- join datasets (short+long+external?)
- treat secondary labels with lower value than 1 (?)
- add folds

### Training

- use rating as sample weight (?)
- use secondary labels with label value < 1 and linear activation (?)
- do something about class imbalance
- do augmentation with mixing of random fragments

---

- use sounscapes data for fine-tuning

### Models

- Baseline: ensemble of 2d and 1d convnets
- Perceiver (?)

## Ideas

- Collect more data, particularly on rare classes
- Use only soundscapes for validation to be closer to test dataset
- Assign default rating value more optimally (is there correllation between length/other stuff and rating?)
- Implement SWA <https://arxiv.org/pdf/1803.05407.pdf>, <https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/>

### Crazy Ideas

- Create synthetic data with GANs

## Links

- "Perceiver: General Perception with Iterative Attention" <https://arxiv.org/pdf/2103.03206.pdf>
- "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" <https://arxiv.org/pdf/2006.10739.pdf>
- <https://www.kaggle.com/stefankahl/birdclef2021-exploring-the-data>
- <http://dcase.community/challenge2018/task-bird-audio-detection-results>
- <http://dcase.community/documents/challenge2018/technical_reports/DCASE2018_Lasseck_76.pdf>
- SWA <https://arxiv.org/pdf/1803.05407.pdf>, <https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/>
