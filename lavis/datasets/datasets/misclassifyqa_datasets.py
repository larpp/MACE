import os
import pandas as pd
from PIL import Image
import torch
import json

from lavis.datasets.datasets.base_dataset import BaseDataset

from lavis.datasets.datasets.vqa_datasets import VQAEvalDataset
from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation.iloc[index]

        return OrderedDict(
            {
                "file": ann["data_path"],
                "answers": ann["answer"],
                "image": sample["image"],
                "gt_label": ann["label"],
                "mispredicted_label": ann["predicted"],
                "category": ann["selected_categories"],
                "caption": ann["caption"],
                "detected_objects": ann["detected_objects"]
            }
        )


class MisclassifyQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, step=2, train_samples_portion="all"):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        
        # CSV 파일을 읽어와서 주석(annotation)으로 사용합니다.
        self.annotation = []
        for ann in ann_paths:
            self.annotation = pd.read_csv(ann, encoding='latin1')
        
        if not ((type(train_samples_portion) == int and train_samples_portion > 0) or train_samples_portion == "all" ):
            raise ValueError("train_samples_portion must be a positive integer or \"all\"")
        if train_samples_portion != "all":
            self.annotation = self.annotation.sample(n=train_samples_portion)
        
        self.step = step
        if self.step == 1:
            # answer_list for vocabulary ranking method (1-step)
            self.answer_list = ["a", "b", "c", "d", "e", "f", "g", "h"]
        elif self.step == 2:
            self.answer_list = None

    def __getitem__(self, index):
        # 주어진 인덱스에서 데이터를 가져옵니다.
        ann = self.annotation.iloc[index]


        # 이미지 경로 설정 및 이미지 로딩
        image_path = os.path.join(self.vis_root, ann["data_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        instruction = self.get_text_input(ann, self.step)

        
        # 질문과 답변에 대해 전처리 수행
        text_input = self.text_processor(instruction)
        if self.step == 1:
            answer = ann["selected_categories"]
            category = ann["selected_categories"]
        elif self.step == 2:
            answer = ann["answer"]
            if 'pred_category' in ann:
                category = ann['pred_category']
            else:
                category = ann["selected_categories"]

        # gt label and mispredicted label
        gt_label = ann["label"]
        mispredicted_label = ann["predicted"]

        caption = ann["caption"]
        objects = ann["detected_objects"]

        return {
            "image": image,
            "text_input": text_input,
            "text_output": answer,
            "gt_label": gt_label,
            "mispredicted_label": mispredicted_label,
            "category": category,
            "caption": caption,
            "objects": objects
        }
        
    def collater(self, samples):
        # 배치에 대한 처리 로직
        image_list, question_list, answer_list, gt_list, mispredicted_list, category_list, caption_list, objects_list = [], [], [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])
            gt_list.append(sample["gt_label"])
            mispredicted_list.append(sample["mispredicted_label"])
            category_list.append(sample["category"])
            caption_list.append(sample["caption"])
            objects_list.append(sample["objects"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
            "gt_label": gt_list,
            "mispredicted_label": mispredicted_list,
            "category": category_list,
            "caption": caption_list,
            "objects": objects_list
        }
    
    @staticmethod
    def get_text_input(sample, step):
        choices = ""
        category = [
            "Visual resemblance",
            "Occlusion or poor visibility",
            "Contextual confusion",
            "Inclusion of predicted label (MIS)",
            "Image corruption or quality issues",
            "Label ambiguity",
            "Mislabeling",
            "Inherent model failure"
        ]

        i = 0
        # 0 -> a, 1 -> b, 2 -> c, 3 -> d, 4 -> e
        for choice in category:
            label = chr(ord('a') + i)
            choices += f"({label}) {choice}\n"
            i += 1
        
        if step == 1:
            text_input = f"""<Image>
You are an AI assistant that analyzes why a classification model misclassified an image.

You are given the following information about a misclassified image:
Caption describing the image: {sample["caption"]}
Detected objects in the image: {sample["detected_objects"]}

Your task is to identify the reason(s) why the model misclassified the image.
Review the image together with the information above and select the most appropriate category or categories from the list below that explain the misclassification.
Category:{choices}
Return only the category letter(s), separated by commas if multiple apply. (e.g., a, b, c)
If you select g, h, or i, do not combine them with other categories. These are standalone categories. (e.g., h)
Do not include any explanation or extra text.

Question: Which category or categories best explain why the model misclassified this image as "{sample["predicted"]}" instead of "{sample["label"]}"? (There may be more than one correct answer. List all applicable answers)
Answer(s): """
            
#             text_input = f"""<Image>
# Caption describing the image: {sample["caption"]}
# Detected objects in the image: {sample["detected_objects"]}
# Category:{choices}
# Question: Which category or categories best explain why the model misclassified this image as "{sample["predicted"]}" instead of "{sample["label"]}"? (There may be more than one correct answer. List all applicable answers)
# Answer(s): """

#             text_input = f"""<Image> System: {{You are tasked with identifying the reason for a misclassification from the given categories based on the provided image. Select one or more categories that apply and output only the category numbers, separated by commas if more than one applies. However note that if f, g, or h is selected, they must not be combined with any other categories. Do not provide any explanation or additional text. Examples of correct outputs: a or a, b, c. Examples of incorrect outputs: (a) or Based in the image, h.}}
# Question: {{The provided image is of a {sample["label"]}, but the model predicted it as a {sample["predicted"]}. Why such a misclassification occurred? (Check if this image is {sample["label"]} before answering)}}
# Category: {{{choices}}}
# Answer(s) (There may be more than one correct answer. List all applicable answers): """
        

        elif step == 2:
            text_input = f"""<Image>
You are an AI assistant that analyzes why a classification model misclassified an image.

You are given the following information about a misclassified image:
Caption describing the image: {sample["caption"]}
Detected objects in the image: {sample["detected_objects"]}
Selected categories: {sample["selected_categories"]}

Your task is to explain why the model misclassified the image, focusing on the selected categories.
Always review all available information to check whether the selected categories properly reflect the true cause of the misclassification.
Be as specific and descriptive as possible, identifying particular objects, background elements, or image characteristics that may have contributed to the misclassification.
Do not simply rephrase the category definitions, and do not include category letters (a, b, etc.) or numeric scores in your answer.

Category: {choices}
Question: Which category or categories best explain why the model misclassified this image as "{sample["predicted"]}" instead of "{sample["label"]}"?
Answer(s): {sample["selected_categories"]}

Question: Why did the model misclassify this image as "{sample["predicted"]}" instead of "{sample["label"]}"?
Answer (in sentece): """

#             text_input = f"""<Image>
# You are an AI assistant that analyzes why a classification model misclassified an image.

# You are given the following information about a misclassified image:
# Caption describing the image: {sample["caption"]}
# Detected objects in the image: {sample["detected_objects"]}

# Your task is to explain why the model misclassified the image.
# Be as specific and descriptive as possible, identifying particular objects, background elements, or image characteristics that may have contributed to the misclassification.

# Question: Why did the model misclassify this image as "{sample["predicted"]}" instead of "{sample["label"]}"?
# Answer (in sentece): """

        return text_input


class MisclassifyQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, step=2, train_samples_portion="all"):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths=[])
        
        self.annotation = []

        # self.annotation = pd.read_csv(ann_paths[0], encoding='latin1')
        self.annotation = pd.read_csv(ann_paths[0])

        self.step = step

        if self.step == 1:
            self.answer_list = ["a", "b", "c", "d", "e", "f", "g", "h"]
        else:
            self.answer_list = None

        # self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]


        image_path = os.path.join(self.vis_root, ann["data_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        instruction = self.get_text_input(ann, self.step)

        text_input = self.text_processor(instruction)
        if self.step == 1:
            answer = ann["selected_categories"]
            category = ann["selected_categories"]
        elif self.step == 2:
            answer = ann["answer"]
            if 'pred_category' in ann:
                category = ann['pred_category']
            else:
                category = ann["selected_categories"]

        gt_label = ann["label"]
        mispredicted_label = ann["predicted"]

        caption = ann["caption"]
        objects = ann["detected_objects"]

        return {
            "image": image,
            "text_input": text_input,
            "text_output": answer,
            "question_id": int(ann["id"]),
            "gt_label": gt_label,
            "mispredicted_label": mispredicted_label,
            "answer": answer,
            "category": category,
            "caption": caption,
            "objects": objects
        }
    
    def collater(self, samples):
        image_list, question_list, answer_list, id_list, gt_list, mispredicted_list, category, caption_list, objects_list = [], [], [], [], [], [], [], [], []
        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])
            id_list.append(sample["question_id"])
            category.append(sample['category'])
            gt_list.append(sample["gt_label"])
            mispredicted_list.append(sample["mispredicted_label"])
            caption_list.append(sample["caption"])
            objects_list.append(sample["objects"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
            "question_id": id_list,
            "gt_label": gt_list,
            "mispredicted_label": mispredicted_list,
            "category" : category,
            "caption": caption_list,
            "objects": objects_list
        }

    @staticmethod
    def get_text_input(sample, step):
        choices = ""
        category = [
            "Visual resemblance",
            "Occlusion or poor visibility",
            "Contextual confusion",
            "Inclusion of predicted label (MIS)",
            "Image corruption or quality issues",
            "Label ambiguity",
            "Mislabeling",
            "Inherent model failure"
        ]

        i = 0
        # 0 -> a, 1 -> b, 2 -> c, 3 -> d, 4 -> e
        for choice in category:
            label = chr(ord('a') + i)
            choices += f"({label}) {choice}\n"
            i += 1

        # system instruction을 넣은 것이 성능이 좀 더 높음
        if step == 1:
            text_input = f"""<Image>
You are an AI assistant that analyzes why a classification model misclassified an image.

You are given the following information about a misclassified image:
Caption describing the image: {sample["caption"]}
Detected objects in the image: {sample["detected_objects"]}

Your task is to identify the reason(s) why the model misclassified the image.
Review the image together with the information above and select the most appropriate category or categories from the list below that explain the misclassification.
Category:{choices}
Return only the category letter(s), separated by commas if multiple apply. (e.g., a, b, c)
If you select g, h, or i, do not combine them with other categories. These are standalone categories. (e.g., h)
Do not include any explanation or extra text.

Question: Which category or categories best explain why the model misclassified this image as "{sample["predicted"]}" instead of "{sample["label"]}"? (There may be more than one correct answer. List all applicable answers)
Answer(s): """
#             text_input = f"""<Image>
# Caption describing the image: {sample["caption"]}
# Detected objects in the image: {sample["detected_objects"]}
# Category:{choices}
# Question: Which category or categories best explain why the model misclassified this image as "{sample["predicted"]}" instead of "{sample["label"]}"? (There may be more than one correct answer. List all applicable answers)
# Answer(s): """
#             text_input = f"""<Image>
# You are an AI assistant that analyzes why a classification model misclassified an image.
# Your task is to identify the reason(s) why the model misclassified the image.
# Review the image and select the most appropriate category or categories from the list below that explain the misclassification.

# Category: (a) Does any object in the image closely resemble the {sample["predicted"]} in shape, color, texture, or other visual features?
# (b) Occlusion or low visibility: Was the {sample["label"]} partially occluded, too small, far away, or otherwise difficult to clearly perceive?
# (c) Contextual confusion: Could background confusion or scene context have misled the model into predicting {sample["label"]} instead of {sample["label"]}?
# (d) Inclusion of the predicted label: Does the image include the {sample["predicted"]}?
# (e) Image corruption or quality issues: Was the image difficult to interpret due to low quality issues like low resolution, blurriness, dark, pixelation, poor lighting, or visual artifacts?
# (f) Abstrac depiction: Dose {sample["label"]} is not real?
# (g) Label ambiguity: The gt and mis refer to the same object or concept with slight variations in terminology, plurality, or specificity
# (h) Mislableing: The {sample["label"]} is actually missing from the image
# (i) Inherent model failure: The {sample["label"]} is clearly present and distinguishable in the image or there are no visual or contextual clues in the image that could justify predicting {sample["predicted"]}

# Return only the category letter(s), separated by commas if multiple apply. (e.g., a, b, c)
# If you select g, h, or i, do not combine them with other categories. These are standalone categories. (e.g., h)
# Do not include any explanation or extra text.

# Question: Which category or categories best explain why the model misclassified this image as "{sample["predicted"]}" instead of "{sample["label"]}"? (There may be more than one correct answer. List all applicable answers)
# Answer(s): """
            
#             text_input = f"""<Image> System: {{You are tasked with identifying the reason for a misclassification from the given categories based on the provided image. Select one or more categories that apply and output only the category numbers, separated by commas if more than one applies. However note that if f, g, or h is selected, they must not be combined with any other categories. Do not provide any explanation or additional text. Examples of correct outputs: a or a, b, c. Examples of incorrect outputs: (a) or Based in the image, h.}}
# Question: {{The provided image is of a {sample["label"]}, but the model predicted it as a {sample["predicted"]}. Why such a misclassification occurred? (Check if this image is {sample["label"]} before answering)}}
# Category: {{{choices}}}
# Answer(s) (There may be more than one correct answer. List all applicable answers): """
#             text_input = f"""Caption describing the image: {sample["caption"]}
# Detected objects in the image: {sample["detected_objects"]}

# Category: (a) Does any object in the image closely resemble the {sample["predicted"]} in shape, color, texture, or other visual features?
# (b) Occlusion or low visibility: Was the {sample["label"]} partially occluded, too small, far away, or otherwise difficult to clearly perceive?
# (c) Contextual confusion: Could background confusion or scene context have misled the
# model into predicting {sample["label"]} instead of {sample["label"]}?
# (d) Inclusion of the predicted label: Does the image include the {sample["predicted"]}?
# (e) Image corruption or quality issues: Was the image difficult to interpret due to low quality issues like blurriness, dark, pixelation, poor lighting, or visual artifacts?
# (f) Abstrac depiction: Dose {sample["label"]} is not real?
# (g) Label ambiguity: The gt and mis refer to the same object or concept with slight variations in terminology, plurality, or specificity
# (h) Mislableing: The {sample["label"]} is actually missing from the image
# (i) Inherent model failure: The {sample["label"]} is clearly present and distinguishable in the image or there are no visual or contextual clues in the image that could justify predicting {sample["predicted"]}

# Question: Which category or categories best explain why the model misclassified this image as "{sample["predicted"]}" instead of "{sample["label"]}"? There may be more than one correct answer. List all applicable answers)
# Answer(s): """
        
        elif step == 2:
            text_input = f"""<Image>
You are an AI assistant that analyzes why a classification model misclassified an image.

You are given the following information about a misclassified image:
Caption describing the image: {sample["caption"]}
Detected objects in the image: {sample["detected_objects"]}
Selected categories: {sample["selected_categories"]}

Your task is to explain why the model misclassified the image, focusing on the selected categories.
Always review all available information to check whether the selected categories properly reflect the true cause of the misclassification.
Be as specific and descriptive as possible, identifying particular objects, background elements, or image characteristics that may have contributed to the misclassification.
Do not simply rephrase the category definitions, and do not include category letters (a, b, etc.) or numeric scores in your answer.

Category: {choices}
Question: Which category or categories best explain why the model misclassified this image as "{sample["predicted"]}" instead of "{sample["label"]}"?
Answer(s): {sample["selected_categories"]}

Question: Why did the model misclassify this image as "{sample["predicted"]}" instead of "{sample["label"]}"?
Answer (in sentece): """
            
#             text_input = f"""<Image>
# You are an AI assistant that analyzes why a classification model misclassified an image.

# You are given the following information about a misclassified image:
# Caption describing the image: {sample["caption"]}
# Detected objects in the image: {sample["detected_objects"]}

# Your task is to explain why the model misclassified the image.
# Be as specific and descriptive as possible, identifying particular objects, background elements, or image characteristics that may have contributed to the misclassification.

# Question: Why did the model misclassify this image as "{sample["predicted"]}" instead of "{sample["label"]}"?
# Answer (in sentece): """

        return text_input