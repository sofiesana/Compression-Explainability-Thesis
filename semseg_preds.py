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

def get_prediction_images(preds, img_names, save_location):
    resized_preds = F.interpolate(preds, (480, 640))
    for i, pred in enumerate(preds):
        normalized_masks = torch.nn.functional.softmax(pred, dim=0).cpu()
        # Get the class prediction by selecting the class index with maximum probability
        class_predict = normalized_masks.argmax(dim=0).numpy()
        # Get the unique classes present in the prediction
        unique_classes = np.unique(class_predict)
        # Define a colormap based on the number of unique classes
        color_map = plt.get_cmap('viridis', len(unique_classes))
        # Apply the colormap to the class prediction directly
        color_image = color_map(class_predict)
        # Squeeze the color_image if it has an extra dimension
        color_image = np.squeeze(color_image)
        # Ensure values are in the correct range (0 to 1)
        color_image = np.clip(color_image, 0, 1)
        # # Display the color image
        im = plt.imshow(color_image)
        plt.axis('off')

        path = os.path.join(RESULTS_ROOT, save_location, img_names[i] + '_pred.png')

        plt.savefig(path)
        im.clear()
        plt.close()


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
    
    PRUNING_METHODS = ['baseline', 'static', 'dynamic', 'pt']

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

                curr_preds = []

                for i, gt_batch in enumerate(test_loader):
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
                    test_loader = DataLoader(test_dataset, batch_size=10, num_workers=8, shuffle=True, pin_memory=True)
                    evaluator = SceneNetEval(
                            device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
                    
                    network_name = f"{dataset}_disparse_{method}_{ratio}"

                    path_to_model = os.path.join(os.environ.get('TMPDIR'), "pruned", method, method+str(model_num), "tmp/results", f"best_{network_name}.pth")
                    torch.cuda.empty_cache()
                    net = load_model(device, 'y', task, path_to_model)
                    net.eval()

                    curr_preds = []

                    for i, gt_batch in enumerate(test_loader):
            
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