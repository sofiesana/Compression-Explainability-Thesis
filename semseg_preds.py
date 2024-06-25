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


def get_prediction_images(preds, img_names, save_location):
    resized_preds = F.interpolate(preds, (480, 640))
    for i, pred in enumerate(preds):
        normalized_masks = torch.nn.functional.softmax(pred, dim=0).cpu()
        class_predict = normalized_masks.argmax(dim=0).numpy()
        unique_classes = np.unique(class_predict)
        color_map = plt.get_cmap('viridis', len(unique_classes))
        color_image = color_map(class_predict)
        color_image = np.squeeze(color_image)
        color_image = np.clip(color_image, 0, 1)

        path = os.path.join(RESULTS_ROOT, save_location, img_names[i] + '_pred.png')

        # Create a new figure for each image to avoid memory issues
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # Display the image
        im = ax.imshow(color_image)
        ax.axis('off')
        
        # Save the image
        path = os.path.join(RESULTS_ROOT, save_location, img_names[i] + '_pred.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        
        # Clear the image to free memory
        im.remove()
        
        # Close the figure to release memory
        plt.close(fig)


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
        '--ratio', type=int, help='0,70,80,90', default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = NYU_v2(DATA_ROOT, 'test')

    method = args.method
    head = args.head
    task = args.task
    irof_version = args.irof
    ratio = args.ratio
    if task == "None":
        task = None        

    model_num_list = [1]
    PRUNING_RATIOS = [ratio]
    PRUNING_METHODS = [method]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    preds = None
    img_name = None
    image = None
    dataset = 'nyuv2'
    all_preds = {}

    for method in PRUNING_METHODS:
        method_preds = {}
        
        for model_num in model_num_list:
            # print(torch.cuda.memory_summary())
            location = method + str(model_num)

            if method == "baseline":

                rslt_path = os.path.join(RESULTS_ROOT, location)
                if not os.path.isdir(rslt_path):
                    os.makedirs(rslt_path)
                
                print("baseline model " + str(model_num))
                test_loader = DataLoader(test_dataset, batch_size=10, num_workers=8, shuffle=False, pin_memory=True)
                evaluator = SceneNetEval(
                        device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
                                    
                network_name = f"{dataset}_{method}"
                path_to_model = os.path.join(os.environ.get('TMPDIR'), method, method + str(model_num), "tmp/results", f"best_{network_name}.pth")

                torch.cuda.empty_cache()
                net = load_model(device, 'n', task, path_to_model)
                # net = SceneNet(TASKS_NUM_CLASS).to(device)
                # net.load_state_dict(torch.load(path_to_model))
                net.eval()

                curr_preds = []

                for i, gt_batch in enumerate(test_loader):
                    if i >= 5:
                        break
                    print("batch:", i)
        
                    net.eval()
                    gt_batch["img"] = Variable(gt_batch["img"]).to(device)
                    if "seg" in gt_batch:
                        gt_batch["seg"] = Variable(gt_batch["seg"]).to(device)
                    if "depth" in gt_batch:
                        gt_batch["depth"] = Variable(gt_batch["depth"]).to(device)
                    if "normal" in gt_batch:
                        gt_batch["normal"] = Variable(gt_batch["normal"]).to(device)

                    preds = net(gt_batch["img"])
                    curr_preds.extend(preds)
                    
                    img_names = gt_batch["name"]
                    image = gt_batch["img"]

                    for i in range(len(image)):
                        img_names[i] = str(img_names[i]).replace('.png', '')
                    
                    get_prediction_images(preds, img_names, location)
                
                method_preds[model_num] = curr_preds

            else:
                for ratio in PRUNING_RATIOS:
                    print(torch.cuda.memory_summary())
                    location = os.path.join(method + str(model_num), str(ratio))

                    rslt_path = os.path.join(RESULTS_ROOT, location)
                    if not os.path.isdir(rslt_path):
                        os.makedirs(rslt_path)
                    
                    print(f"{method} model {model_num} ratio {ratio}")
                    test_loader = DataLoader(test_dataset, batch_size=10, num_workers=8, shuffle=False, pin_memory=True)
                    evaluator = SceneNetEval(
                            device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
                    
                    network_name = f"{dataset}_disparse_{method}_{ratio}"

                    path_to_model = os.path.join(os.environ.get('TMPDIR'), "pruned", method, method+str(model_num), "tmp/results", f"best_{network_name}.pth")
                    torch.cuda.empty_cache()
                    net = load_model(device, 'y', task, path_to_model)
                    net.eval()

                    curr_preds = []

                    for i, gt_batch in enumerate(test_loader):
                        if i >= 5:
                            break

                        net.eval()
                        gt_batch["img"] = Variable(gt_batch["img"]).to(device)
                        if "seg" in gt_batch:
                            gt_batch["seg"] = Variable(gt_batch["seg"]).to(device)
                        if "depth" in gt_batch:
                            gt_batch["depth"] = Variable(gt_batch["depth"]).to(device)
                        if "normal" in gt_batch:
                            gt_batch["normal"] = Variable(gt_batch["normal"]).to(device)

                        preds = net(gt_batch["img"])
                        curr_preds.extend(preds)
                        
                        img_names = gt_batch["name"]
                        image = gt_batch["img"]

                        for i in range(len(image)):
                            img_names[i] = str(img_names[i]).replace('.png', '')
                        
                        get_prediction_images(preds, img_names, location)

                    method_preds[model_num] = curr_preds

                    # Clear the cache between different pruning ratios
                    del net
                    torch.cuda.empty_cache()
                    gc.collect()
            
            with open(os.path.join(RESULTS_ROOT, method + str(model_num) + '_preds.pkl'), 'wb') as file:
                pickle.dump(method_preds, file)

        all_preds[method] = method_preds
    
    with open(os.path.join(RESULTS_ROOT, 'preds.pkl'), 'wb') as file:
        pickle.dump(all_preds, file)