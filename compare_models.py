import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from dataloaders import *
import matplotlib.pyplot as plt
from scene_net import *

from evaluation import SceneNetEval


from config_nyuv2 import *

PRUNING_METHODS = ["baseline", "pt", "static"]
NUM_MODELS = 3
PRUNING_RATIOS = [50, 70, 80, 90]

RESULTS_ROOT = os.path.join(os.environ.get('TMPDIR'), 'results')

baseline_results = {}
model_results = {}

def plot_metrics(results, metric_name):
    plt.figure(figsize=(12, 8))
    for method, method_results in results.items():
        for model_num, metrics in method_results.items():
            if method == "baseline":
                x = [0]
                y = [metrics[metric_name]]
            else:
                x = PRUNING_RATIOS
                y = [metrics[ratio][metric_name] for ratio in PRUNING_RATIOS]
            plt.plot(x, y, marker='o', label=f"{method} model {model_num}")

    plt.xlabel("Pruning Ratio (%)" if method != "baseline" else "Baseline")
    plt.ylabel(metric_name)
    plt.title(f"Comparison of {metric_name} across models and pruning methods")
    plt.legend()
    plt.grid(True)
    plt.show()

def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    dataset = 'nyuv2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = SceneNet(TASKS_NUM_CLASS).to(device)
    test_dataset = NYU_v2(DATA_ROOT, 'test')

    for method in PRUNING_METHODS:
            method_results = {}
            for model_num in range(1, NUM_MODELS+1):

                if method == "baseline":
                    print("baseline model " + str(model_num))
                    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=True, pin_memory=True)
                    evaluator = SceneNetEval(
                            device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
                                        
                    network_name = f"{dataset}_{method}"
                    path_to_model = os.path.join(os.environ.get('TMPDIR'), method, method + str(model_num), "tmp/results", f"best_{network_name}.pth")
                    net.load_state_dict(torch.load(path_to_model))
                    net.eval()
                    res = evaluator.get_final_metrics(net, test_loader)
                    print(res)

                    method_results[model_num] = res

                else:
                    for ratio in PRUNING_RATIOS:
                        print(f"{method} model {model_num} ratio {ratio}")
                        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=True, pin_memory=True)
                        evaluator = SceneNetEval(
                                device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
                        
                        network_name = f"{dataset}_{method}_{ratio}"

                        path_to_model = os.path.join(os.environ.get('TMPDIR'), "pruned", method, method+str(model_num), "tmp/results", f"best_{network_name}.pth")
                        net.eval()
                        res = evaluator.get_final_metrics(net, test_loader)
                        print(res)

                        if model_num not in method_results:
                            method_results[model_num] = {}

                        method_results[model_num][ratio] = res
            
            model_results[method] = method_results

    
    # Save the results
    results_path = os.path.join(RESULTS_ROOT, 'model_results.pkl')
    save_results(model_results, results_path)

    # Plot the comparison of mIoU and pixel accuracy
    plot_metrics(model_results, 'mIoUs', os.path.join(RESULTS_ROOT, 'mIoUs_comparison.png'))
    plot_metrics(model_results, 'pixelAccs', os.path.join(RESULTS_ROOT, 'pixelAccs_comparison.png'))

                        
            
    
    