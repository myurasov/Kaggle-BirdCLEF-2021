rm -rf _data _work ; cli/download_data.sh && \
jupyter nbconvert --to notebook --execute notebooks/convert_n_data.ipynb && \
cli/prepare_short.run.py --min_rating 3 --max_from_clip 10 --no_rating_value 3 --rectify_class_balance 0 --sample_with_stride 5 --out_csv short-D.csv && \
cli/prepare_long.run.py --split_multilabel 1 --out_csv long-D.csv && \
cli/create_dataset.run.py --in_csvs short-D.csv long-D.csv --out dataset-D.pickle --secondary_label_p 0.33 --rectify_class_balance 2.5 0.25 --geofilter all-500mi-last_5y-1mo_tolerance --folds 7 && \
cli/cache_waves.py --dataset dataset-D.pickle && \
cli/prepare_short.run.py --min_rating 3 --max_from_clip 0 --no_rating_value 3 --rectify_class_balance 0 --sample_with_stride 5 --sample_with_detection_csv /app/res/n_nocall_predictions.csv.gz --out_csv short-F.csv && \
cli/prepare_long.run.py --split_multilabel 1 --out_csv long-F.csv && \
cli/create_dataset.run.py --in_csvs n_nocall.csv short-F.csv long-F.csv --out dataset-F.pickle --secondary_label_p 0.33 --rectify_class_balance 2.5 0.25 --geofilter all-500mi-last_5y-1mo_tolerance --folds 7 && \
cli/cache_waves.py --dataset dataset-F.pickle && \
cli/train_fe.run.py --run F1_d_xa --dataset dataset-F.pickle --lr 0.001 --lr_patience 5 --samples_per_epoch 128000 --model msg_enb4_imagenet_xauxenc332 --amp 1 --val_fold 0.1 --batch 32 --preload_val_data 1 --multiprocessing 8x4 --epochs 500  --weight_by_rareness 0 --weight_by_rating 1 --monitor_metric val_f1_score
