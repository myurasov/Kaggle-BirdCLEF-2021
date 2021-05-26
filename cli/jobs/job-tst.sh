#!/bin/bash

/cmd &

cli/prepare_short.run.py --min_rating 3 --max_from_clip 0 --no_rating_value 3 --rectify_class_balance 0 --sample_with_stride 5 --sample_with_detection_csv /app/res/n_nocall_predictions.csv.gz --out_csv short-TST.csv

cli/prepare_long.run.py --split_multilabel 1 --out_csv long-TST.csv

cli/create_dataset.run.py --in_csvs n_nocall.csv short-TST.csv long-TST.csv --out dataset-TST.pickle --secondary_label_p 0.33 --rectify_class_balance 2.5 0.25 --geofilter all-500mi-last_5y-1mo_tolerance --folds 7

cli/cache_waves.py --dataset dataset-TST.pickle

cli/train_fe.run.py --run TST --dataset dataset-TST.pickle --lr 0.001 --lr_patience 3 --lr_factor 0.5 --samples_per_epoch 1280 --model msg_enb4_imagenet_xauxenc332 --amp 1 --val_fold 0.001 --batch 32 --preload_val_data 0 --multiprocessing 4x4 --epochs 5  --weight_by_rareness 0 --weight_by_rating 1 --monitor_metric val_f1_score

# copy the result
cp -r _work/models /result/

sleep infinity
