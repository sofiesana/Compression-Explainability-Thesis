from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import gc
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

# PRUNING_METHODS = ["baseline", "pt", "static", "dynamic"]
NUM_MODELS = 3
PRUNING_RATIOS = [50, 70, 80, 90]

def run_irof_seg(model, test_loader, location):

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
                                    num_classes=40, 
                                    task = task
                                    )

            class_mask_float = get_binary_mask(preds, class_category)
            attributions = get_attributions(model, image, class_category, class_mask_float)
            
            valid_indices = []

            # i is img index in batch
            for i, img_seg in enumerate(torch.argmax(preds, axis=1)):
                if class_category not in img_seg:
                    # print(class_name, " not in image ", str(img_names[i]))
                    pass
                if np.all((attributions[i] == 0)):
                    # print("attributions all zero for image ", str(img_names[i]))
                    pass
                if class_category in img_seg and not np.all((attributions[i] == 0)):
                    valid_indices.append(i)

            # print("valid:", valid_indices)


            if valid_indices:

                path = os.path.join(RESULTS_ROOT, location, class_name)
                if not os.path.isdir(path):
                    os.makedirs(path)
                # print("Directory '% s' created" % path)

                y_batch = preds.argmax(axis=1)
                
                reduced_image_names = np.array(img_names)[valid_indices]
                y_batch = y_batch[valid_indices]
                x_batch = image[valid_indices]
                a_batch = attributions[valid_indices]
                reduced_preds = preds[valid_indices]

                get_resized_binary_mask(reduced_image_names, reduced_preds, class_name, class_category, location)

                class_mask_float = get_binary_mask(reduced_preds, class_category)
                attributions = get_attributions(model, x_batch, class_category, class_mask_float)

                get_gradcam_image(reduced_image_names, a_batch, x_batch, location, class_name)
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
    

    for category in class_scores.keys():
        class_name = sem_idx_to_class[category]
        plot_all_irof_curves(class_histories[category], location, class_name)
        plot_avg_irof_curve(class_histories[category], location, class_name)

    return class_scores, class_histories

def run_irof_sn(model, test_loader, location):
    all_scores = []
    all_histories = []

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
        
        irof = quantus.IROF(segmentation_method="slic",
                                perturb_baseline=irof_version,
                                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                return_aggregate=False,
                                class_category=None,
                                class_name=None,
                                num_classes=40,
                                task = task
                                )

        attributions = get_attributions(model, image)
        get_sn_image(img_names, preds, location)

        valid_indices = []

        for i, att in enumerate(attributions):
            # print(att)
            if not np.all((att == 0)):
                # print("valid")
                valid_indices.append(i)

        if valid_indices:
            # print(valid_indices)
            reduced_image_names = np.array(img_names)[valid_indices]
            y_batch = preds[valid_indices]
            x_batch = gt_batch["img"][valid_indices]
            a_batch = attributions[valid_indices]

            get_gradcam_image(reduced_image_names, a_batch, x_batch, location)

            scores, histories = irof(model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=device)
            
            if scores is not None:
                all_scores.extend(scores)
                all_histories.extend(histories)
        
    plot_all_irof_curves(all_histories, location)
    plot_avg_irof_curve(all_histories, location)

    return all_scores, all_histories


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='IROF Evaluation')
    parser.add_argument(
        '--head', type=str, help='mtl model head: all, ', default="all")
    parser.add_argument(
        '--method', type=str, help='pruning method to evaluate: baseline, pt, dynamic, static', default="baseline")
    parser.add_argument(
        '--task', type=str, help='seg, sn', default="None")
    parser.add_argument(
        '--irof', type=str, help='mean, uniform', default="mean")
    parser.add_argument(
        '--model_num', type=int, help='1,2,3', default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = NYU_v2(DATA_ROOT, 'test')

    method = args.method
    head = args.head
    task = args.task
    irof_version = args.irof
    model_number = args.model_num
    if task == "None":
        task = None        

    if model_number is None:
        print("No model number given, running on all models")
        model_num_list = [1,2,3]
    else:
        model_num_list = [model_number]
    
    PRUNING_METHODS = [method]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    preds = None
    img_name = None
    image = None
    dataset = 'nyuv2'
    all_scores = {}
    all_histories = {}

    for method in PRUNING_METHODS:
        method_histories = {}
        method_scores = {}
        
        for model_num in model_num_list:
            # print(torch.cuda.memory_summary())
            location = method + str(model_num)

            if method == "baseline":

                rslt_path = os.path.join(RESULTS_ROOT, location)
                if not os.path.isdir(rslt_path):
                    os.makedirs(rslt_path)
                
                print("baseline model " + str(model_num))
                test_loader = DataLoader(test_dataset, batch_size=10, num_workers=8, shuffle=True, pin_memory=True)
                evaluator = SceneNetEval(
                        device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
                                    
                network_name = f"{dataset}_{method}"
                path_to_model = os.path.join(os.environ.get('TMPDIR'), method, method + str(model_num), "tmp/results", f"best_{network_name}.pth")

                torch.cuda.empty_cache()
                net = load_model(device, 'n', task, path_to_model)
                # net = SceneNet(TASKS_NUM_CLASS).to(device)
                # net.load_state_dict(torch.load(path_to_model))
                net.eval()

                if task == 'seg':
                    scores, histories = run_irof_seg(net, test_loader, location)
                elif task == 'sn':
                    scores, histories = run_irof_sn(net, test_loader, location)
                    
                else:
                    print("task not recognized")
                    break
                
                method_histories[model_num] = histories
                method_scores[model_num] = scores

            else:
                for ratio in PRUNING_RATIOS:
                    print(torch.cuda.memory_summary())
                    location = os.path.join(method + str(model_num), str(ratio))

                    rslt_path = os.path.join(RESULTS_ROOT, location)
                    if not os.path.isdir(rslt_path):
                        os.makedirs(rslt_path)
                    
                    print(f"{method} model {model_num} ratio {ratio}")
                    test_loader = DataLoader(test_dataset, batch_size=10, num_workers=8, shuffle=True, pin_memory=True)
                    evaluator = SceneNetEval(
                            device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
                    
                    # net = SceneNet(TASKS_NUM_CLASS).to(device)

                    # for module in net.modules():
                    #     # Check if it's basic block
                    #     if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
                    #         module = prune.identity(module, 'weight')

                    # need to actually retrieve the model
                    network_name = f"{dataset}_disparse_{method}_{ratio}"

                    path_to_model = os.path.join(os.environ.get('TMPDIR'), "pruned", method, method+str(model_num), "tmp/results", f"best_{network_name}.pth")
                    torch.cuda.empty_cache()
                    net = load_model(device, 'y', task, path_to_model)
                    net.eval()

                    if task == 'seg':
                        scores, histories = run_irof_seg(net, test_loader, location)
                    elif task == 'sn':
                        scores, histories = run_irof_sn(net, test_loader, location)
                    else:
                        print("task not recognized")
                        break
                    
                    if model_num not in method_scores:
                        method_scores[model_num] = {}
                        method_histories[model_num] = {}

                    method_scores[model_num][ratio] = scores
                    method_histories[model_num][ratio] = histories
                
                    # Clear the cache between different pruning ratios
                    del net
                    torch.cuda.empty_cache()
                    gc.collect()
            
            with open(os.path.join(RESULTS_ROOT, method + str(model_num) + '_histories.pkl'), 'wb') as file:
                pickle.dump(method_histories, file)

            with open(os.path.join(RESULTS_ROOT, method + str(model_num) + '_scores.pkl'), 'wb') as file:
                pickle.dump(method_scores, file)

        all_scores[method] = method_scores
        all_histories[method] = method_histories
    
    # print(all_scores)

    with open(os.path.join(RESULTS_ROOT, 'histories.pkl'), 'wb') as file:
        pickle.dump(all_histories, file)
    
    with open(os.path.join(RESULTS_ROOT,'scores.pkl'), 'wb') as file:
        pickle.dump(all_scores, file)