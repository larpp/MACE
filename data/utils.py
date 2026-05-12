import re
import json
import os


def pre_caption(caption, max_chars=512):
    # 문장 나누기
    sentences = caption.split('. ')

    # 문장이 하나뿐이면 처음부터 512자까지
    if len(sentences) == 1:
        truncated_caption = caption[:max_chars]
    else:
        # 첫 번째 문장을 제외하고 나머지 문장들 이어붙이기
        remaining_caption = '. '.join(sentences[1:])
        truncated_caption = remaining_caption[:max_chars]

        # 특수 문자 제거 및 공백 정리
    caption = re.sub(r"([.!\"()*#:;~])", ' ', truncated_caption.lower())
    caption = re.sub(r"\s{2,}", ' ', caption)
    caption = caption.rstrip('\n').strip(' ')

    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def convert_to_coco_format(
    test_file, 
    coco_gt_file='gt.json', 
    coco_results_file='results.json', 
    question_id_key="question_id", 
    gt_ans_key="gt_ans", 
    pred_ans_key="pred_ans",
    indent=4
):
    """
    Converts a test.json file into COCO format for caption evaluation.

    Args:
        test_file (str): Path to the input test file (JSON format).
        coco_gt_file (str): Path to save the COCO format ground truth file.
        coco_results_file (str): Path to save the COCO format results file.
        question_id_key (str): Key for the unique identifier in test_file (default: "question_id").
        gt_ans_key (str): Key for the ground truth answers in test_file (default: "gt_ans").
        pred_ans_key (str): Key for the predicted answers in test_file (default: "pred_ans").
    """
    root_path = os.path.dirname(test_file)

    coco_gt_path = os.path.join(root_path, coco_gt_file)
    coco_results_path = os.path.join(root_path, coco_results_file)

    # Load the input test file
    with open(test_file, "r") as f:
        data = json.load(f)

    # Initialize the COCO format dictionaries
    annotations = {"annotations": [], "images": []}
    results = []

    # Process each item in the test data
    for item in data:
        question_id = item[question_id_key]
        gt_ans = pre_caption(item[gt_ans_key])
        pred_ans = pre_caption(item[pred_ans_key])

        # Add ground truth to annotations
        annotations["annotations"].append({
            "id" : question_id,
            "image_id": question_id,  # Use question_id as image_id
            "caption": gt_ans  # Ground truth answer
        })

        annotations["images"].append({
            "id" : question_id
        })

        # Add predictions to results
        results.append({
            "id" : question_id,
            "images" : question_id,
            "image_id": question_id,  # Use question_id as image_id
            "caption": pred_ans  # Predicted answer
        })

    # Save the annotations and results as JSON files
    with open(coco_gt_path, "w") as f:
        json.dump(annotations, f, indent=indent)

    with open(coco_results_path, "w") as f:
        json.dump(results, f, indent=indent)

    print(f"COCO format conversion completed!\n- Ground Truth: {coco_gt_file}\n- Results: {coco_results_file}")

    return coco_gt_path, coco_results_path


# Example usage:
# convert_to_coco_format("test.json", "coco_gt.json", "coco_results.json")


from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def coco_caption_eval(coco_gt_root, results_file):    
    # create coco object and coco_result object
    coco = COCO(coco_gt_root)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval


def jaccard_similarity_from_json(data):
    """
    JSON 형식 데이터를 입력받아 Jaccard Similarity를 계산.

    Parameters:
        data (dict): JSON 데이터로, 각 항목은 question_id, pred_ans, gt_ans로 구성됨.

    Returns:
        dict: 각 question_id에 대한 Jaccard Similarity와 평균 Similarity.
    """
    # similarities = {}
    total_similarity = 0

    for item in data:
        # question_id = item["question_id"]
        pred_ans = set([i.strip() for i in item["pred_ans"].split(",")])
        gt_ans = set([i.strip() for i in item["gt_ans"].split(",")])

        # Jaccard Similarity 계산
        intersection = len(pred_ans.intersection(gt_ans))
        union = len(pred_ans.union(gt_ans))
        similarity = intersection / union if union != 0 else 0.0

        # similarities[question_id] = similarity
        total_similarity += similarity

    # 평균 Similarity 계산
    avg_similarity = total_similarity / len(data) if data else 0.0

    return avg_similarity



def precision_recall_f1_from_json(data):
    """
    JSON 형식 데이터를 입력받아 Precision, Recall, F1 Score의 평균을 계산.

    Parameters:
        data (list of dict): 각 항목은 question_id, pred_ans, gt_ans로 구성됨.

    Returns:
        dict: Precision, Recall, F1 Score의 평균 값.
    """
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for item in data:
        pred_ans = set([i.strip() for i in item["pred_ans"].split(",")])
        gt_ans = set([i.strip() for i in item["gt_ans"].split(",")])

        # True Positives, False Positives, False Negatives
        tp = len(pred_ans.intersection(gt_ans))
        fp = len(pred_ans - gt_ans)
        fn = len(gt_ans - pred_ans)

        # Precision, Recall, F1 Score 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # 평균 계산
    num_items = len(data)

    average_precision = total_precision / num_items if num_items > 0 else 0.0
    average_recall = total_recall / num_items if num_items > 0 else 0.0
    average_f1 = total_f1 / num_items if num_items > 0 else 0.0

    return average_precision, average_recall, average_f1


def subset_accuracy_from_json(data):
    """
    JSON 형식 데이터를 입력받아 Subset Accuracy의 평균을 계산.

    Parameters:
        data (list of dict): 각 항목은 question_id, pred_ans, gt_ans로 구성됨.

    Returns:
        float: Subset Accuracy의 평균 값.
    """
    correct_predictions = 0

    for item in data:
        pred_ans = set([i.strip() for i in item["pred_ans"].split(",")])
        gt_ans = set([i.strip() for i in item["gt_ans"].split(",")])

        # Subset Accuracy 계산 (예측이 정답과 완전히 일치하는 경우만 1로 계산)
        if pred_ans == gt_ans:
            correct_predictions += 1

    # 평균 계산
    num_items = len(data)
    average_subset_accuracy = correct_predictions / num_items if num_items > 0 else 0.0

    return average_subset_accuracy


def hamming_loss_from_json(data):
    """
    JSON 형식 데이터를 입력받아 Hamming Loss의 평균을 계산.

    Parameters:
        data (list of dict): 각 항목은 question_id, pred_ans, gt_ans로 구성됨.

    Returns:
        float: Hamming Loss의 평균 값.
    """
    total_hamming_loss = 0
    total_labels = 0

    for item in data:
        pred_ans = set([i.strip() for i in item["pred_ans"].split(", ")])
        gt_ans = set([i.strip() for i in item["gt_ans"].split(", ")])

        # False Positive와 False Negative 계산
        fp_and_fn = len(pred_ans.symmetric_difference(gt_ans))
        total_hamming_loss += fp_and_fn
        total_labels += len(pred_ans.union(gt_ans))

    # 평균 Hamming Loss 계산
    average_hamming_loss = total_hamming_loss / total_labels if total_labels > 0 else 0.0

    return average_hamming_loss
