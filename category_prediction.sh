# Train
python -m torch.distributed.run --nproc_per_node=4 --master_port=29505 train.py \
--cfg-path lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_misclassifyqa_64.yaml

# Evaluation
# python -m torch.distributed.run --nproc_per_node=4 --master_port=29505 train.py \
# --cfg-path lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_misclassifyqa_64_eval.yaml
