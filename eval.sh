python3 main.py \
--coco_path /root/detr/data/dil1600_400  --num_workers 8 \
--no_aux_loss --eval --valid_batch_size 8 \
--num_queries 600 --num_classes 300 \
--delta 14 --target_short 1600 --target_short_min 980 \
--global_short_max 1240 --global_short_min 980 \
--long_short_ratio 1.414 --global_local_ratio 0.5 \
--local_threshold 0.7 --local_width_min 0.6 --local_height_min 0.4 \
--pretrained /root/detr/data/pretrained/checkpoint2402_0.5528.pth
#--resume /root/detr/pred_attr_9/checkpoint0034.pth