# Code for BirdCLEF 2021 Kaggle Competition

<https://www.kaggle.com/c/birdclef-2021>

## Plan

### Dataset Creation

- read `train_metadata.csv`
- select only labels with good rating
- slice clips into fragments with `stride` and `length` cut fragments based on detection model
- convert coordinates into orthogonal basis,  bin them to a coarse grid (10x10?)
- add date coarsened up to season (month?, 1/8 of y?)
- add secondary labels

- read `train_soundscape_labels.csv`
- add date coarsened date
- add coarsened coordinates
- assume rating is '5' (?)
- assume all labels are primary (?)

- add folds

### Training

- use rating as sample weight (?)
- use secondary labels with label value < 1 and linear activation (?)
- do something about class imbalance
- do augmentation with mixing of random fragments

### Models

- Baseline: ensemble of 2d and 1d convnets
- Perceiver (?)

## Ideas

- Collect more data, particularly on rare classes

### Crazy Ideas

- Create synthetic data with GANs

## Links

- "Perceiver: General Perception with Iterative Attention" <https://arxiv.org/pdf/2103.03206.pdf>
- "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" <https://arxiv.org/pdf/2006.10739.pdf>
