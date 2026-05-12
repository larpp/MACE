# Train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=2 --master_port=29505 train.py \
--cfg-path lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_misclassifyqa_64_2.yaml

# Evaluation
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=2 --master_port=29505 train.py \
# --cfg-path lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_misclassifyqa_64_2_eval.yaml