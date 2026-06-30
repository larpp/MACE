# MACE: A Misclassification-Aware Framework for Image Classification Evaluation

## Overview

Recent advances in computer vision have raised interest not only in how models achieve high accuracy but also in why they fail. Explainable AI (XAI) aims to interpret model decisions, yet existing evaluation metrics and methods rarely address the causes of incorrect predictions. Even highly accurate models can misclassify images, and understanding these failures is crucial for improving reliability and guiding model selection. We question the conventional evaluation paradigm that measures success solely by accuracy and propose a framework that systematically identifies and explains misclassifications, extending existing evaluation practices to incorporate these insights. For each test image, our method analyzes the root cause of the error, enabling a more comprehensive and realistic assessment of model performance. While previous XAI studies mainly highlight visual features influencing predictions, they do not explain why a model made an incorrect decision in human-understandable terms. Leveraging large language models (LLMs), we introduce an LLM-based framework that categorizes misclassification causes into eight categories, including visual resemblance, occlusion, label ambiguity, and mislabeling. We implement a rule-based decision-making framework in which GPT-4o acts as an annotator to estimate confidence scores and generate explanations for each category. We further construct a dataset of misclassification explanations and instruction-tune multimodal LLMs such as InstructBLIP and LLaVA-1.5. Experiments show that our model effectively categorizes and explains failure patterns, providing a new perspective on benchmarking classification performance.

## Contents

- [Install](#install)
- [Train](#train)
- [Evaluation](#evaluation)

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

We train our model using 4 A6000 GPUs with 48GB memory.

## Evaluation

