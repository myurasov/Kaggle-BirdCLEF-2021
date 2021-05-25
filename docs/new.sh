rm -rf _data _work ; cli/download_data.sh && \
jupyter nbconvert --to notebook --execute notebooks/convert_n_data.ipynb && \


cli/prepare_short.run.py --min_rating 3 --max_from_clip 0 --no_rating_value 3 --rectify_class_balance 0 --sample_with_stride 5 --sample_with_detection_csv /app/res/n_nocall_predictions.csv.gz --out_csv short-E.csv && \
cli/prepare_long.run.py --split_multilabel 1 --out_csv long-E.csv && \
cli/create_dataset.run.py --in_csvs n_nocall.csv short-E.csv long-E.csv --out dataset-E.pickle --secondary_label_p 0.33 --rectify_class_balance 2.5 0.25 --geofilter all-500mi-last_5y-1mo_tolerance --folds 7 && \
cli/cache_waves.py --dataset dataset-E.pickle && \

cli/train_fe.run.py --run E1_d_xa_augv2 --dataset dataset-E.pickle --lr 0.001 --lr_patience 5 --samples_per_epoch 128000 --model msg_enb4_imagenet_xauxenc332 --amp 1 --val_fold 0.1 --batch 32 --preload_val_data 1 --multiprocessing 8x4 --epochs 500  --weight_by_rareness 0 --weight_by_rating 1 --monitor_metric val_f1_score --aug v2

