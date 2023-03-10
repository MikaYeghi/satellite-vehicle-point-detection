import os
import torch
import itertools
import numpy as np
import pandas as pd
import detectron2.utils.comm as comm
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from detectron2.data import DatasetCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

import pdb

class COCOPointEvaluator(DatasetEvaluator):
    def __init__(self,
        dataset_name,
        logger,
        accept_radius=6,
        tasks=None,
        distributed=True,
        output_dir=None,
        plot_results=False,
        FP_FN_analysis=False,
        eval_margin_size=6,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,
    ):
        # Settings
        self.logger = logger
        self.dataset_name = dataset_name
        self.accept_radius = accept_radius
        self.tasks = tasks
        self.distributed = distributed
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "results")
        self.max_dets_per_image = max_dets_per_image
        self.use_fast_impl = use_fast_impl
        self.kpt_oks_sigmas = kpt_oks_sigmas
        self.allow_cached_coco = allow_cached_coco
        self.plot_results = plot_results
        
        # Evaluation 
        self.predictions = []
        self.metadata = DatasetCatalog.get(dataset_name)
        self.FP_FN_analysis = FP_FN_analysis
        self.eval_margin_size = eval_margin_size # Predictions and GT-s that are close to the image border are discarded
        
        # Create the save directory
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def reset(self):
        self.predictions = []
        
    def process(self, inputs, outputs):
        """
        Predictions are stored in COCO bounding box format, i.e.: (x_min, y_min, width, height).
        """
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
        """
        GT locations are stored in (x_min, y_min, x_max, y_max) format, while predictions are processed
        into COCO format, i.e. (x_min, y_min, width, height). This is accounted for whenever computing
        the bounding box center locations.
        """
        # Synchronize distributed processes
        if self.distributed:
            comm.synchronize()
            predictions = comm.gather(self.predictions, dst=0)
            self.predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        
        # Sort the predictions and GT lists by image_id
        self.metadata = sorted(self.metadata, key=lambda d: d['image_id'])
        self.predictions = sorted(self.predictions, key=lambda d: d['image_id'])
        filenames = []
        TP_list = []
        scores_list = []
        
        # Remove predictions and GT points which are within the specified margin
        self.apply_margin()
        
        # FP-FN-analysis
        if self.FP_FN_analysis:
            FN_list = []
            total_fn_count = 0
        
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
            # If there are no predictions, just update the GT count
            if len(gt_data['annotations']) == 0:
                for j in range(len(pred_data['instances'])):
                    self.predictions[i]['instances'][j]['TP'] = False
                    TP_list.append(False)
                    scores_list.append(pred_data['instances'][j]['score'])
                continue
            elif len(pred_data['instances']) == 0:
                total_gt_count += len(gt_data['annotations'])
                continue
            else:
                total_gt_count += len(gt_data['annotations'])
            
            # Otherwise, assess which prediction is TP
            pred_locations = torch.tensor(
                [[bbox['bbox'][0] + 0.5 * bbox['bbox'][2], bbox['bbox'][1] + 0.5 * bbox['bbox'][3]] for bbox in pred_data['instances']],
                dtype=torch.float
            )
            gt_locations = torch.tensor(
                [0.5 * (bbox['bbox'][:2] + bbox['bbox'][2:]) for bbox in gt_data['annotations']],
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
            
            # Record FNs
            if self.FP_FN_analysis:
                FN_info = self.metadata[i].copy()
                FN_idxs = [i for i in range(len(gt_locations)) if i not in TP_idxs]
                FN_count = len(FN_idxs)
                FN_info['FN_idxs'] = FN_idxs
                FN_info['FN_count'] = FN_count
                FN_list.append(FN_info)
                total_fn_count += len(FN_idxs)
        
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
        evaluation_info["F1"] = evaluation_info.apply(lambda row : 2 * row['Precision'] * row['Recall'] / (row['Precision'] + row['Recall'] + 1e-6), axis=1)
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
        self.logger.info(f"RESULTS:\nAverage precision: {round(100 * ap, 2)}%.\nOptimal confidence level: {round(100 * optimal_confidence, 2)}%.\nBest F1 score: {round(100 * F1_max, 2)}%.")
        ########### AVERAGE PRECISION CALCULATION END ###########
        
        # Construct the results dictionary
        results = {
            "AP": ap,
            "Best F1 score": F1_max,
            "Optimal confidence": optimal_confidence
        }
        
        # Plot inference results
        if self.plot_results:
            self.logger.info("Plotting the results")
            self.save_results(self.predictions.copy(), self.metadata.copy(), optimal_confidence, precision_list, recall_list)
        
        # FP-FN-analysis
        if self.FP_FN_analysis:
            # Sort the list of FNs by the FN ratio
            image_counter = 0
            
            # Save directory
            FP_FN_save_dir = os.path.join(self.results_dir, "FP-FN-analysis")
            Path(FP_FN_save_dir).mkdir(parents=True, exist_ok=True)
            
            # Sort the images by the number of FNs in a decreasing order
            FN_list = sorted(FN_list, key=lambda d: d['FN_count'], reverse=True)
            
            # Plot the results
            self.logger.info("Plotting FN examples.")
            for image_info in tqdm(FN_list):
                image = Image.open(image_info['file_name']).convert('RGB')
                
                # Stop plotting if there're no images left with FNs
                if len(image_info['FN_idxs']) == 0:
                    break
                
                # Plot the original image
                fig, ax = plt.subplots()
                ax.imshow(image)
                
                # Plot GT data
                for idx in range(len(image_info['annotations'])):
                    is_FN = idx in image_info['FN_idxs']
                    gt_x = 0.5 * (image_info['annotations'][idx]['bbox'][0] + image_info['annotations'][idx]['bbox'][2])
                    gt_y = 0.5 * (image_info['annotations'][idx]['bbox'][1] + image_info['annotations'][idx]['bbox'][3])
                    if is_FN:
                        color = 'r'
                    else:
                        color = 'g'
                    ax.plot(gt_x, gt_y, color + 'o')
                
                # Save the figure
                fig.savefig(os.path.join(FP_FN_save_dir, f"image_{image_counter}.png"), dpi=150)
                
                image_counter += 1
                plt.close(fig)
        
        return results
    
    def save_results(self, predictions, metadata, confidence_threshold, precision_list, recall_list):
        # Remove predictions which have confidence score lower than threshold
        predictions_ = []
        for prediction in predictions:
            prediction_ = prediction.copy()
            prediction_['instances'] = []
            for instance in prediction['instances']:
                if instance['score'] >= confidence_threshold:
                    prediction_['instances'].append(instance)
            predictions_.append(prediction_)
        predictions = predictions_
        
        # Sort the predictions and GT lists by image_id
        metadata = sorted(metadata, key=lambda d: d['image_id'])
        predictions = sorted(predictions, key=lambda d: d['image_id'])
        
        # Save directory
        save_dir = os.path.join(self.results_dir, "predictions")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Plot the images
        for i in tqdm(range(len(predictions))):
            assert metadata[i]['image_id'] == predictions[i]['image_id']
            
            # Skip empty images
            if len(metadata[i]['annotations']) == 0 and len(predictions[i]['instances']) == 0:
                continue
            
            # Retrieve the image
            image = Image.open(metadata[i]['file_name'])
            file_name = metadata[i]['file_name'].split('/')[-1]
            
            # Initialize a plt figure
            fig, ax = plt.subplots()
            
            # Get x and y coordinates of the predictions, and whether they are TP or FP
            pred_xs = [instance['bbox'][0] + 0.5 * instance['bbox'][2] for instance in predictions[i]['instances']]
            pred_ys = [instance['bbox'][1] + 0.5 * instance['bbox'][3] for instance in predictions[i]['instances']]
            TPs = [instance['TP'] for instance in predictions[i]['instances']]
            
            # Get the ground-truth points
            gt_xs = [0.5 * (annotation['bbox'][0] + annotation['bbox'][2]) for annotation in metadata[i]['annotations']]
            gt_ys = [0.5 * (annotation['bbox'][1] + annotation['bbox'][3]) for annotation in metadata[i]['annotations']]
            
            # Assign colors to points depending on the TP value
            pred_colors = ['green' if TP else 'red' for TP in TPs]
            gt_colors = ['blue' for _ in range(len(metadata[i]['annotations']))]
            
            # Plot the image
            ax.imshow(image)
            
            # Plot the ground-truth colors
            ax.scatter(gt_xs, gt_ys, c=gt_colors)
            
            # Plot the predicted colors
            ax.scatter(pred_xs, pred_ys, c=pred_colors)
            
            # Save the plot
            fig.savefig(os.path.join(save_dir, file_name))
            
            # Clear plt
            del fig, ax
            plt.close()
    
        # Plot the P-R curve
        fig, ax = plt.subplots()
        ax.plot(precision_list, recall_list)
        PR_curve_save_path = os.path.join(self.results_dir, "PR_curve.png")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curve")
        fig.savefig(PR_curve_save_path, dpi=150)
    
    def apply_margin(self):
        """
        This function removes all GT and predicted points from self.metadata and self.predictions which fall into the restricted margin.
        The restricted margin is specified with the self.eval_margin_size parameter, which is typically half of the fake bounding box size, and is used
        to remove truncated vehicles (both GT and predicted).
        """
        for global_counter, (gt_data, pred_data) in enumerate(zip(self.metadata, self.predictions)):
            assert gt_data['image_id'] == pred_data['image_id'], "GT and prediction image ID-s are different!"
            new_gt_data = {}
            new_pred_data = {}

            # Clean the GT data
            gt_valid_idxs = []
            for idx, annotation in enumerate(gt_data['annotations']):
                gt_bbox = annotation['bbox']
                gt_point = np.array([0.5 * (gt_bbox[0] + gt_bbox[2]), 0.5 * (gt_bbox[1] + gt_bbox[3])])
                if self.eval_margin_size < gt_point[0] < gt_data['width'] - self.eval_margin_size and self.eval_margin_size < gt_point[1] < gt_data['height'] - self.eval_margin_size:
                    gt_valid_idxs.append(idx)
            for k, v in gt_data.items():
                if k != 'annotations':
                    new_gt_data[k] = v
                else:
                    new_gt_data[k] = []
                    for idx in gt_valid_idxs:
                        new_gt_data[k].append(gt_data[k][idx])

            # Clean the predictions
            pred_valid_idxs = []
            for idx, instance in enumerate(pred_data['instances']):
                pred_bbox = instance['bbox']
                pred_point = np.array([pred_bbox[0] + 0.5 * pred_bbox[2], pred_bbox[1] + 0.5 * pred_bbox[3]])
                if self.eval_margin_size < pred_point[0] < gt_data['width'] - self.eval_margin_size and self.eval_margin_size < pred_point[1] < gt_data['height'] - self.eval_margin_size:
                    pred_valid_idxs.append(idx)
            for k, v in pred_data.items():
                if k != 'instances':
                    new_pred_data[k] = v
                else:
                    new_pred_data[k] = []
                    for idx in pred_valid_idxs:
                        new_pred_data[k].append(pred_data[k][idx])
        
            # Save the clean GT and predictions
            self.metadata[global_counter] = new_gt_data
            self.predictions[global_counter] = new_pred_data