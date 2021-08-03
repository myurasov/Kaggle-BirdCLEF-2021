# Code for BirdCLEF 2021 Kaggle Competition

<https://www.kaggle.com/c/birdclef-2021>

## Running

### Starting Jupyter Lab and TensorBoard

`$ docker/docker-forever.sh [--gpus='device=###|all'] [--jupyter_port=####|8888] [--tensorboard_port=####|6006]`

### Shell

`$ docker/docker.sh [--gpus='device=###|all'] bash`

## Workflow

### Create geofilter

```sh
cli/get_flybies.run.py --dataset dataset-all-m1.pickle --miles 500 --time_tolerance_months 1 --only_months 1 --last_n_years 5 > docs/flybies-all-500mi-last_5y-1mo_tolerance.txt
```

### Prepare data

```sh
cli/prepare_short.run.py --min_rating 3 --max_from_clip 10 --no_rating_value 3 --rectify_class_balance 0 --sample_with_stride 5 --out_csv short-C.csv && \
cli/prepare_long.run.py --split_multilabel 1 --out_csv long-C.csv && \
cli/create_dataset.run.py --in_csvs short-X.csv long-X.csv --out dataset-X.pickle --secondary_label_p 0.33 --rectify_class_balance 1 --geofilter all-500mi-last_5y-1mo_tolerance --folds 7 && \
cli/cache_waves.py --dataset dataset-X.pickle
```

### Train

```
cli/train_fe.run.py --run X --dataset dataset-X.pickle --lr 0.001 --lr_patience 5 --samples_per_epoch 128000 --model msg_enb4_imagenet_noxdense --amp 1 --val_fold 0.15 --batch 64 --preload_val_data 1 --multiprocessing 4x4 --epochs 500  --weight_by_rareness 0 --monitor_metric val_f1_score
```

### Upload code to Kaggle

Configure credentials in the `kaggle.json` polaced in the root of the project.

```sh
cli/kaggle_upload_repo.sh
```
