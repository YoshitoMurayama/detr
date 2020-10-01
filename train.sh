CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8  --use_env main.py \
--coco_path /root/detr/data/dil1600_400 --num_workers 8 \
--num_queries 600 --num_classes 300 \
--epochs 3000 --lr 1e-5 --lr_drop 3000 \
--batch_size 3 --valid_batch_size 8 \
--delta 14 --target_short 1600 --target_short_min 980 \
--global_short_max 1240 --global_short_min 980 \
--long_short_ratio 1.414 --global_local_ratio 0.5 \
--local_threshold 0.7 --local_width_min 0.6 --local_height_min 0.4 \
--resume /root/detr/pred_attr_9/checkpoint0034.pth --save_threshold 0.542 \
--output_dir /root/detr/pred_attr_9

