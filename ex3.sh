mv /home/InstructBLIP_PEFT/input/misclassifyqa/test.csv /home/InstructBLIP_PEFT/input/misclassifyqa/test_final.csv
mv /home/InstructBLIP_PEFT/input/misclassifyqa/images /home/InstructBLIP_PEFT/input/misclassifyqa/images_origin
mv /home/InstructBLIP_PEFT/input/misclassifyqa/images_ /home/InstructBLIP_PEFT/input/misclassifyqa/images


mv /home/InstructBLIP_PEFT/input/misclassifyqa/other_dataset_csv/caltech_vit.csv /home/InstructBLIP_PEFT/input/misclassifyqa/test.csv
mv /home/InstructBLIP_PEFT/input/misclassifyqa/blip_caltech_vit /home/InstructBLIP_PEFT/input/misclassifyqa/images/data/ImageNet/val
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 --master_port=29505 train.py \
--cfg-path lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_misclassifyqa_64_eval.yaml
mv /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/2025*/ /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/real_cal_vit

mv /home/InstructBLIP_PEFT/input/misclassifyqa/test.csv /home/InstructBLIP_PEFT/input/misclassifyqa/other_dataset_csv/caltech_vit.csv
mv /home/InstructBLIP_PEFT/input/misclassifyqa/images/data/ImageNet/val /home/InstructBLIP_PEFT/input/misclassifyqa/blip_caltech_vit


mv /home/InstructBLIP_PEFT/input/misclassifyqa/other_dataset_csv/cifar100_vit.csv /home/InstructBLIP_PEFT/input/misclassifyqa/test.csv
mv /home/InstructBLIP_PEFT/input/misclassifyqa/blip_cifar100_vit /home/InstructBLIP_PEFT/input/misclassifyqa/images/data/ImageNet/val
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 --master_port=29505 train.py \
--cfg-path lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_misclassifyqa_64_eval.yaml
mv /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/2025*/ /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/real_cifar_vit

mv /home/InstructBLIP_PEFT/input/misclassifyqa/test.csv /home/InstructBLIP_PEFT/input/misclassifyqa/other_dataset_csv/cifar100_vit.csv
mv /home/InstructBLIP_PEFT/input/misclassifyqa/images/data/ImageNet/val /home/InstructBLIP_PEFT/input/misclassifyqa/blip_cifar100_vit


mv /home/InstructBLIP_PEFT/input/misclassifyqa/other_dataset_csv/objectnet_vit.csv /home/InstructBLIP_PEFT/input/misclassifyqa/test.csv
mv /home/InstructBLIP_PEFT/input/misclassifyqa/blip_objectnet_vit /home/InstructBLIP_PEFT/input/misclassifyqa/images/data/ImageNet/val
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 --master_port=29505 train.py \
--cfg-path lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_misclassifyqa_64_eval.yaml
mv /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/2025*/ /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/real_obj_vit

mv /home/InstructBLIP_PEFT/input/misclassifyqa/test.csv /home/InstructBLIP_PEFT/input/misclassifyqa/other_dataset_csv/objectnet_vit.csv
mv /home/InstructBLIP_PEFT/input/misclassifyqa/images/data/ImageNet/val /home/InstructBLIP_PEFT/input/misclassifyqa/blip_objectnet_vit


mv /home/InstructBLIP_PEFT/input/misclassifyqa/test_final.csv /home/InstructBLIP_PEFT/input/misclassifyqa/test.csv
mv /home/InstructBLIP_PEFT/input/misclassifyqa/images /home/InstructBLIP_PEFT/input/misclassifyqa/images_
mv /home/InstructBLIP_PEFT/input/misclassifyqa/images_origin /home/InstructBLIP_PEFT/input/misclassifyqa/images


# ---------------------------------------------------------------

# # NR
# python -m torch.distributed.run --nproc_per_node=2 train.py \
# --cfg-path lavis/projects/instructblip/train/misclassifyqa/0.yaml

# mv /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/2025*/ /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/neurips_nr

# python -m torch.distributed.run --nproc_per_node=2 train.py \
# --cfg-path lavis/projects/instructblip/train/misclassifyqa/10_eval.yaml

# mv /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/2025*/ /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/neurips_nr_result


# # TNC
# python -m torch.distributed.run --nproc_per_node=2 train.py \
# --cfg-path lavis/projects/instructblip/train/misclassifyqa/3_eval.yaml

# mv /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/2025*/ /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/neurips_tnc_result


# # NTNC
# python -m torch.distributed.run --nproc_per_node=2 train.py \
# --cfg-path lavis/projects/instructblip/train/misclassifyqa/6_eval.yaml

# mv /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/2025*/ /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/neurips_ntnc_result


# # NTC
# python -m torch.distributed.run --nproc_per_node=2 train.py \
# --cfg-path lavis/projects/instructblip/train/misclassifyqa/0_eval.yaml

# mv /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/2025*/ /home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/neurips_ntc_result

