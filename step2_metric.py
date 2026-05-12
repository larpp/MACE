from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def coco_caption_eval(coco_gt_root, results_file):
    coco = COCO(coco_gt_root)
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval


if __name__=='__main__':
    coco_caption_eval('/home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/20250204200/result/gt.json', '/home/InstructBLIP_PEFT/output/results/misclassifyqa/misclassifyqa_64/20250204200/result/results.json')
