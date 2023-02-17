import torch
import numpy as np
import pandas as pd
from detectron2.data import DatasetCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

import pdb

class COCOPointEvaluator(DatasetEvaluator):
    def __init__(self,
        dataset_name,
        accept_radius=6,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,
    ):
        # Settings
        self.dataset_name = dataset_name
        self.accept_radius = accept_radius
        self.tasks = tasks
        self.distributed = distributed
        self.output_dir = output_dir
        self.max_dets_per_image = max_dets_per_image
        self.use_fast_impl = use_fast_impl
        self.kpt_oks_sigmas = kpt_oks_sigmas
        self.allow_cached_coco = allow_cached_coco
        
        # Evaluation 
        self.predictions = []
        self.metadata = DatasetCatalog.get(dataset_name)
    
    def reset(self):
        self.predictions = []
        
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to('cpu')
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to('cpu')
            if len(prediction) > 1:
                self.predictions.append(prediction)
    
    def evaluate(self):
        # Sort the predictions and GT lists by image_id
        self.metadata = sorted(self.metadata, key=lambda d: d['image_id'])
        self.predictions = sorted(self.predictions, key=lambda d: d['image_id'])
        filenames = []
        TP_list = []
        scores_list = []
        
        # Assert the lengths of the predictions and GT lists are equal
        assert len(self.metadata) == len(self.predictions)
        
        # Total count of GT points -- used for computing the recall
        total_gt_count = 0
        
        # Label each prediction's TP as either true or false
        for i in range(len(self.metadata)):
            gt_data = self.metadata[i]
            pred_data = self.predictions[i]
            assert gt_data['image_id'] == pred_data['image_id']
            n_preds = len(pred_data['instances'])
            
            # Record file names
            filenames = filenames + [gt_data['file_name']] * n_preds
            
            # If there are no GT points, label all predictions as FP
            if len(gt_data['annotations']) == 0:
                for j in range(len(pred_data['instances'])):
                    self.predictions[i]['instances'][j]['TP'] = False
                    TP_list.append(False)
                    scores_list.append(pred_data['instances'][j]['score'])
                continue
            else:
                total_gt_count += len(gt_data['annotations'])
            
            # Otherwise, assess which prediction is TP
            pred_locations = torch.tensor(
                [bbox['bbox'][:2] for bbox in pred_data['instances']],
                dtype=torch.float
            )
            gt_locations = torch.tensor(
                [bbox['bbox'][:2] for bbox in gt_data['annotations']],
                dtype=torch.float
            )
            distances = torch.cdist(gt_locations, pred_locations)
            
            # Find the closest predictions
            idxs = torch.argmin(distances, dim=1)
            distances = torch.gather(distances, 1, idxs.unsqueeze(1))
            
            # Get TP indices
            TP_idxs = torch.where(distances <= self.accept_radius)[0]
            
            # Record TPs
            for j in range(len(pred_data['instances'])):
                is_TP = j in TP_idxs
                self.predictions[i]['instances'][j]['TP'] = is_TP
                TP_list.append(is_TP)
                scores_list.append(pred_data['instances'][j]['score'])
            
        # Construct the evaluation info dataframe
        evaluation_info = pd.DataFrame(
            list(zip(filenames, scores_list, TP_list)), 
            columns=["File names", "Confidence", "TP"]
        )
        
        # Sort the dataframe in descending confidence order
        evaluation_info = evaluation_info.sort_values(by=['Confidence'], ascending=False)

        # Create the FP column
        evaluation_info["FP"] = evaluation_info.apply(lambda row : 1 - row['TP'], axis=1)

        # Create the accumulated TP and FP columns
        evaluation_info["Acc_TP"] = evaluation_info['TP'].cumsum()
        evaluation_info["Acc_FP"] = evaluation_info['FP'].cumsum()

        # Create the "Precision" and "Recall" columns
        evaluation_info["Precision"] = evaluation_info.apply(lambda row : row["Acc_TP"] / (row["Acc_TP"] + row["Acc_FP"]), axis=1)
        evaluation_info["Recall"] = evaluation_info.apply(lambda row : row["Acc_TP"] / total_gt_count, axis=1)

        # Create the "F1" score column and extract the optimal confidence
        evaluation_info["F1"] = evaluation_info.apply(lambda row : 2 * row['Precision'] * row['Recall'] / (row['Precision'] + row['Recall']), axis=1)
        optimal_confidence = evaluation_info.loc[evaluation_info['F1'].idxmax()]['Confidence']
        F1_max = evaluation_info.loc[evaluation_info['F1'].idxmax()]['F1']

        ########### AVERAGE PRECISION CALCULATION START ###########
        recall_list = evaluation_info["Recall"].to_numpy()
        precision_list = evaluation_info["Precision"].to_numpy()
        recall_list = np.concatenate(([0.], recall_list, [1.]))
        precision_list = np.concatenate(([1.], precision_list, [0.]))

        # Compute and the average precision
        for i in range(precision_list.size - 1, 0, -1):
            precision_list[i - 1] = np.maximum(precision_list[i - 1], precision_list[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(recall_list[1:] != recall_list[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((recall_list[i + 1] - recall_list[i]) * precision_list[i + 1])
        print(f"Average precision: {round(100 * ap, 2)}%.")
        ########### AVERAGE PRECISION CALCULATION END ###########
        
        return None