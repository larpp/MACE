# python -m torch.distributed.run --nproc_per_node=1 evaluate.py \
# --cfg-path lavis/projects/instructblip/eval/misclassifyqa_eval.yaml

# bash run_scripts/instructblip/train/run_finetune_instructblip_experiments.sh misclassifyqa 64

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
--cfg-path lavis/projects/instructblip/eval/misclassifyqa_eval.yaml