from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from dataloaders import *
from scene_net import *
from evaluation import *
from irof_utils import *
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from prune_utils import *
import tqdm
import pickle
import argparse
import os
import sys

# Get the value of the TMPDIR environment variable
tmpdir = os.environ.get('TMPDIR')

#Importing
quantus_dir = os.path.join(tmpdir, 'quantus/Quantus-Thesis-Version')
sys.path.insert(0, quantus_dir)

import quantus

PRUNING_METHODS = ["baseline", "pt", "static", "dynamic"]
NUM_MODELS = 3
PRUNING_RATIOS = [50, 70, 80, 90]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IROF Evaluation')
    parser.add_argument(
        '--head', type=str, help='mtl model head: all, ', default="all")
    parser.add_argument(
        '--pruned', type=str, help='is the model pruned?: y, n', default="n")
    parser.add_argument(
        '--task', type=str, help='seg, sn', default="None")
    parser.add_argument(
        '--irof', type=str, help='mean, uniform', default="mean")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = NYU_v2(DATA_ROOT, 'test')
    test_loader = DataLoader(test_dataset, batch_size=10,
                             num_workers=8, shuffle=True, pin_memory=True)

    pruned = args.pruned
    head = args.head
    task = args.task
    irof_version = args.irof
    if task == "None":
        task = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, pruned, task)

    all_preds = []
    preds = None
    img_name = None
    image = None
    class_scores = {}
    class_histories = {}

    sem_idx_to_class = {idx: cls for (idx, cls) in enumerate(CLASS_NAMES)}
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(CLASS_NAMES)}

    for i, gt_batch in enumerate(test_loader):
        
        model.eval()
        gt_batch["img"] = Variable(gt_batch["img"]).to(device)
        if "seg" in gt_batch:
            gt_batch["seg"] = Variable(gt_batch["seg"]).to(device)
        if "depth" in gt_batch:
            gt_batch["depth"] = Variable(gt_batch["depth"]).to(device)
        if "normal" in gt_batch:
            gt_batch["normal"] = Variable(gt_batch["normal"]).to(device)

        preds = model(gt_batch["img"])
        
        img_names = gt_batch["name"]
        image = gt_batch["img"]

        for i in range(len(image)):
            img_names[i] = str(img_names[i]).replace('.png', '')
        
        for class_category in range(TASKS_NUM_CLASS[0]):

            class_name = sem_idx_to_class[class_category]

            irof = quantus.IROF(segmentation_method="slic",
                                    perturb_baseline=irof_version,
                                    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                    return_aggregate=False,
                                    class_category=class_category,
                                    class_name=class_name,
                                    num_classes=40
                                    )

            class_mask_float = get_binary_mask(preds, class_category)
            attributions = get_attributions(model, class_category, class_mask_float, image)
            
            valid_indices = []

            # i is img index in batch
            for i, img_seg in enumerate(torch.argmax(preds, axis=1)):
                if class_category not in img_seg:
                    print(class_name, " not in image ", str(img_names[i]))
                if np.all((attributions[i] == 0)):
                    print("attributions all zero for image ", str(img_names[i]))
                if class_category in img_seg and not np.all((attributions[i] == 0)):
                    valid_indices.append(i)

            print("valid:", valid_indices)


            if valid_indices:

                path = os.path.join(RESULTS_ROOT, class_name)
                if not os.path.isdir(path):
                    os.mkdir(path)
                print("Directory '% s' created" % path)

                y_batch = preds.argmax(axis=1)
                
                reduced_image_names = np.array(img_names)[valid_indices]
                y_batch = y_batch[valid_indices]
                x_batch = image[valid_indices]
                a_batch = attributions[valid_indices]
                reduced_preds = preds[valid_indices]

                get_resized_binary_mask(reduced_image_names, reduced_preds, class_name, class_category)

                class_mask_float = get_binary_mask(reduced_preds, class_category)
                attributions = get_attributions(model, class_category, class_mask_float, x_batch)

                get_gradcam_image(reduced_image_names, a_batch, x_batch, class_name)

                scores, histories = irof(model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=device)

                if scores is not None:
                    if class_category not in class_scores.keys():
                        class_scores[class_category] = []
                        class_histories[class_category] = []

                    class_scores[class_category].extend(scores)
                    class_histories[class_category].extend(histories)
    
    mean_aoc = {}

    for category in class_scores.keys():
        class_name = sem_idx_to_class[category]
        mean_aoc[category] = np.mean(np.array(class_scores[category]))
        plot_all_irof_curves(class_histories[category], class_name)
        plot_avg_irof_curve(class_histories[category], class_name)

    with open(os.path.join(RESULTS_ROOT, 'histories.pkl'), 'wb') as file:
        pickle.dump(class_histories, file)
    
    with open(os.path.join(RESULTS_ROOT,'scores.pkl'), 'wb') as file:
        pickle.dump(class_scores, file)