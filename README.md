# MACE: A Misclassification-Aware Framework for Image Classification Evaluation

## Overview

<p align=center>
<img width="900" alt="Image" src="https://github.com/user-attachments/assets/739ef6f8-58fb-4fbb-89b3-86cd29461c04" />
</p>

This work introduces MACE, a misclassification-aware framework for evaluating image classification models beyond accuracy. Instead of only counting wrong predictions, MACE diagnoses why each error occurs and explains it in natural language. It defines eight failure categories, including visual resemblance, occlusion, contextual confusion, inclusion of predicted label, image quality issues, label ambiguity, mislabeling, and inherent model failure. Using GPT-4o as a structured annotator, the framework assigns error categories, generates detailed explanations, and builds a misclassification explanation dataset for instruction-tuning multimodal LLMs. The results show that MACE can reveal dataset-related issues, model-specific weaknesses, and more realistic evaluation metrics for reliable model selection.

## Contents

- [Install](#install)
- [Train](#train)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Install

1. Clone this repository and navigate to MACE folder.
```shell
https://github.com/larpp/MACE.git
cd MACE
```

2. Install Package
```
pip install -r requirements.txt
```

## Train

Training proceeds in two stages: (1) category prediction: where the model identifies the cause of a misclassification; (2) explanation generation: where the model explains the reason for the error in natural language based on the predicted category.

We train our model using 4 A6000 GPUs with 48GB memory.

### Dataset

For detailed instructions on how to create the dataset, please refer to [this page](https://github.com/larpp/MACE/tree/main/generation_dataset).

Datasets must be placed in the location specified in the file ```lavis/config/datasets/misclassify/default.yaml```.

```yaml
# lavis/config/datasets/misclassify/default.yaml
datasets:
  misclassifyqa:
    data_type: images

    build_info:
      # Be careful not to append minus sign (-) before split to avoid itemizing
      annotations:
        train:
          storage: /home/MACE/input/misclassifyqa/train.csv
        test:
          storage: /home/MACE/input/misclassifyqa/test.csv
      images:
        storage: /home/MACE/input
        train:
          storage: /home/MACE/input
        test:
          storage: /home/MACE/input
```

In this case, dataset json files (```train.csv``` and ```test.csv```) should be located at ```/input/misclassifyqa```.<br>
Image files should be located according to the ```input/{data_path}``` column in the CSV file.

### 1. Category Prediction

```shell
sh category_prediction.sh
```

### 2. Generate Explanantion

The training is resumed from the checkpoint obtained in the previous category prediction step, allowing the model to generate explanations based on the learned misclassification categories.<br>
Set the previous checkpoint path in ```lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_miclassifyqa_64_2_eval.yaml```

```yaml
# lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_miclassifyqa_64_2_eval.yaml
model:
  load_finetuned: True
  finetuned: "/home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/2026/checkpoint_4.pth"
```

Then run the following code.

```shell
sh explanation_generation.sh
```

## Evaluation

You can evaluate with this command.

### 1. Category Prediction

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=29505 train.py \
--cfg-path lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_misclassifyqa_64_eval.yaml
```

### 2. Generate Explanantion

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=29505 train.py \
--cfg-path lavis/projects/instructblip/train/misclassifyqa/finetune_instructblip_misclassifyqa_64_2_eval.yaml
```

## Comparison with GPT-4o and GPT-5.2

<p align=center>
<img width="700" alt="Image" src="https://github.com/user-attachments/assets/4c34deef-7c71-43ea-aeef-d633a318cf5a" />
</p>

## Misclassification-Aware Metric

<p align=center>
<img width="600" alt="Image" src="https://github.com/user-attachments/assets/52fa73b2-1341-45d0-8819-b1587ff420e2" />
</p>

## Citation

```

```

## Acknowledgement

- [InstructBLIP PEFT](https://github.com/AttentionX/InstructBLIP_PEFT): the codebase we built upon
