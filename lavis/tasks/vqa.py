"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import torch
import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_vizwiz import VQA_Vizwiz
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.common.vqa_tools.vqa_eval_vizwiz import VQAEval_Vizwiz
from lavis.tasks.base_task import BaseTask


@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
        num_captions=1,
        do_sample=False,
        step=1,
        top_p=0.9,
        temperature=1.0,
        length_penalty=1.0
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt
        self.num_captions = num_captions
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature
        self.length_penalty = length_penalty

        self.answer_list = None

        self.step = step

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)
        num_captions = run_cfg.get("num_captions", 1)
        do_sample = run_cfg.get("do_sample", False)
        top_p = run_cfg.get("top_p", 0.9)
        temperature = run_cfg.get("temperature", 1.0)
        length_penalty = run_cfg.get("length_penalty", 1.0)
        step = run_cfg.get("step", 2)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
            num_captions=num_captions,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            length_penalty=length_penalty,
            step=step
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )

            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics

@registry.register_task("gqa")
class GQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        
        for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        TODO: add other evaluation metrics for GQA
        """

        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]

            # if self.inference_method == "generate":
            pred = vqa_tool.processPunctuation(pred)
            pred = vqa_tool.processDigitArticle(pred)

            vqa_acc = 1 if pred == gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics
        

@registry.register_task("aok_vqa")
class AOKVQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
        )

        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["direct_answers"]

        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            pred_qa_pairs.append(
                {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer}
            )

        return pred_qa_pairs

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Implementing accuracy computation for AOKVQA, see
        https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py#L45 for details.
        """
        # TODO add evaluation for multi-choice

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            pred = res["pred_ans"]
            gt_ans = res["gt_ans"]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / 3.0)

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

    @dist_utils.main_process
    def _save_result_leaderboard(self, results):
        """
        Saving the results in the format required for leaderboard evaluation.

        [TODO] add support for multi-choice.
        """
        result_leaderboard = dict()
        for res in results:
            result_leaderboard[res["question_id"]] = {
                "direct_answer": res["pred_ans"],
                "multiple_choice": "",
            }

        result_file = registry.get_path("result_dir") + "_leaderboard.json"

        with open(result_file, "w") as f:
            json.dump(result_leaderboard, f)

        logging.info(f"Saved results for leaderboard evaluation at {result_file}")

@registry.register_task("scienceqa")
class ScienceQATask(VQATask):

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        # make predicted answers
        # answers = model.predict_answers(
        #     samples=samples,
        #     answer_list=self.answer_list,
        #     inference_method=self.inference_method,
        #     num_beams=self.num_beams,
        #     max_len=self.max_len,
        #     min_len=self.min_len,
        #     num_ans_candidates=self.num_ans_candidates,
        #     prompt=self.prompt,
        # )
        candidates = []
        if not isinstance(samples, list):
            i = 0
            for choice in samples["choices"][0]:
                label = chr(ord('a') + i)
                candidates.append(f"({label}) {choice}")
                i += 1
        else:
            candidates = samples['choices']
        
        answers = model.predict_class(
            samples=samples,
            candidates=candidates,
            n_segments=1,
        )
        pred_qa_pairs = []


        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        # img_names = samples["image_name"]
        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            # ques_id = int(ques_id)
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        # print(val_result[:5])
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_scienceqa_result",
            remove_duplicate="",
        )
        if split_name == 'val':
            metrics = self._report_metrics(result_file=result_file, split=split_name)
        else:
            metrics = None 
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        # scienceQA metric is easy
        # just check if the predicted answer is the ground truth answer

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            # if res["gt_ans"] is None:
            #     # prepare test results for leaderboard evaluation
            #     self._save_result_leaderboard(results)
            #     return

            pred = res["pred_ans"]
            gt_ans = [res["gt_ans"]]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / len(gt_ans))

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), f"log.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics
      

@registry.register_task("vizwiz")
class VizWizTask(VQATask):

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets
    
    def train_step(self, model, samples):
        txtout = []
        for i in range(len(samples["text_output"])):
            txtout.extend(samples["text_output"][i])
        sample_final = {"image" : torch.cat([samples["image"]] *10 ), "text_input": samples["text_input"]*10, "text_output": txtout}
    
        output = model(sample_final)
        loss_dict = {}
        for k,v in output.items():
            if "loss" in k:
                loss_dict[k] = v

        return output["loss"], loss_dict
    
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        img_names = samples["image_name"]
        for answer, img_name in zip(answers, img_names):
            # ques_id = int(ques_id)
            pred_qa_pairs.append({"image": img_name, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="",
        )
        if split_name == 'val':
            metrics = self._report_metrics(result_file=result_file, split=split_name)
        else:
            metrics = None 
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official Vizwiz evaluation script to report metrics.
        """
        metrics = {}
        print(result_file)
         
        annFile = "../../../input/disk-50gb/vizwiz/annotations/" + split + ".json"
        vqa = VQA_Vizwiz(annFile)
        vqa_result = VQA_Vizwiz(result_file)

        # create vqaEval object by taking vqa and vqaRes
        # n is precision of accuracy (number of places after decimal), default is 2
        vqa_scorer = VQAEval_Vizwiz(vqa, vqa_result, n=2)
        logging.info("Start VQA Vizwiz evaluation.")
        vqa_scorer.evaluate()

        # print accuracies
        overall_acc = vqa_scorer.accuracy["overall"]
        metrics["agg_metrics"] = overall_acc

        logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
        logging.info("Per Answer Type Accuracy is the following:")

        for ans_type in vqa_scorer.accuracy["perAnswerType"]:
            logging.info(
                "%s : %.02f"
                % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
            )
            metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            print(f"wrote result on {f}")
            f.write(json.dumps(metrics) + "\n")

        return metrics

    
@registry.register_task("iconqa")
class IconQATask(VQATask):

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    # def train_step(self, model, samples):
    #     txtout = []
    #     for i in range(len(samples["text_output"])):
    #         txtout.extend(samples["text_output"][i])
    #     sample_final = {"image" : torch.cat([samples["image"]] *10 ), "text_input": samples["text_input"]*10, "text_output": txtout}
    
    #     output = model(sample_final)
    #     loss_dict = {}
    #     for k,v in output.items():
    #         if "loss" in k:
    #             loss_dict[k] = v

    #     return output["loss"], loss_dict
    
        # answers = model.predict_answers(
        #     samples=samples,
        #     answer_list=self.answer_list,
        #     inference_method=self.inference_method,
        #     num_beams=self.num_beams,
        #     max_len=self.max_len,
        #     min_len=self.min_len,
        #     num_ans_candidates=self.num_ans_candidates,
        #     prompt=self.prompt,
        # )
        # pred_qa_pairs = []

        # question_id = samples["question_id"]
        # img_names = samples["image_name"]
        # for answer, img_name in zip(answers, img_names):
        #     # ques_id = int(ques_id)
        #     pred_qa_pairs.append({"image": img_name, "answer": answer})

        # return pred_qa_pairs
    

    def valid_step(self, model, samples):
        # make predicted answers
        # answers = model.predict_answers(
        #     samples=samples,
        #     answer_list=self.answer_list,
        #     inference_method=self.inference_method,
        #     num_beams=self.num_beams,
        #     max_len=self.max_len,
        #     min_len=self.min_len,
        #     num_ans_candidates=self.num_ans_candidates,
        #     prompt=self.prompt,
        # )
        candidates = []
        if not isinstance(samples, list):
            i = 0
            for choice in samples["choices"][0]:
                label = chr(ord('a') + i)
                candidates.append(f"({label}) {choice}")
                i += 1
        else:
            candidates = samples['choices']
        
        answers = model.predict_class(
            samples=samples,
            candidates=candidates,
            n_segments=1,
        )
        pred_qa_pairs = []


        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        # img_names = samples["image_name"]
        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            # ques_id = int(ques_id)
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        # print(val_result[:5])
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_iconqa_result",
            remove_duplicate="",
        )
        if split_name == 'val':
            metrics = self._report_metrics(result_file=result_file, split=split_name)
        else:
            metrics = None 
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        # IconeQA metric is easy
        # just check if the predicted answer is the ground truth answer

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            # if res["gt_ans"] is None:
            #     # prepare test results for leaderboard evaluation
            #     self._save_result_leaderboard(results)
            #     return

            pred = res["pred_ans"]
            gt_ans = [res["gt_ans"]]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / len(gt_ans))

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), f"log.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics
    

@registry.register_task("misclassifyqa")
class MisclassifyQATask(VQATask):
    
    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        for dataset in datasets.values():
            for split in dataset:
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        # make predicted answers
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
            use_nucleus_sampling=self.do_sample,
            num_captions=self.num_captions,
            top_p=self.top_p,
            temperature=self.temperature,
            length_penalty=self.length_penalty
        )
        # answers = model.generate(
        #     samples=samples,
        #     num_beams=self.num_beams,
        #     max_length=self.max_len,
        #     min_length=self.min_len
        # )
        pred_qa_pairs = []


        question_id = samples["question_id"]
        gt_answers = samples["text_output"]
        # img_names = samples["image_name"]
        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            # ques_id = int(ques_id)
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer})

        return pred_qa_pairs


    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_misclassifyqa_result",
            remove_duplicate="question_id", # remove_duplicate=""
        )
        
        if split_name == 'val':
            metrics = self._report_accuracy_metrics(result_file=result_file, split=split_name)
        elif split_name == 'test':
            # 1-step : Accuracy metric
            if self.step == 1:
                metrics = self._report_accuracy_metrics(result_file=result_file, split=split_name)

            # 2-step : Caption metric
            elif self.step == 2:
                metrics = self._report_caption_metrics(result_file=result_file, split=split_name)
        else:
            metrics = None 
        return metrics
    

    @dist_utils.main_process
    def _report_accuracy_metrics(self, result_file, split):
        from data import utils

        results = json.load(open(result_file, "r"))
        
        jaccard_similarity = utils.jaccard_similarity_from_json(results)
        precision, recall, f1 = utils.precision_recall_f1_from_json(results)
        emr = utils.subset_accuracy_from_json(results)
        hamming_loss = utils.hamming_loss_from_json(results)


        metrics = {
            "jaccard_similarity": f"{jaccard_similarity:.3f}",
            "precision": f"{precision:.3f}",
            "recall": f"{recall:.3f}",
            "f1_score": f"{f1:.3f}",
            "exact_match_ratio": f"{emr:.3f}",
            "hamming_loss": f"{hamming_loss:.3f}"
        }

        with open(
            os.path.join(registry.get_path("output_dir"), f"log.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics


    @dist_utils.main_process
    def _report_caption_metrics(self, result_file, split):
        from data import utils

        coco_gt_path, results_file = utils.convert_to_coco_format(result_file)

        coco_test = utils.coco_caption_eval(coco_gt_path, results_file)

        return coco_test