import os
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
import pickle
from prune_utils import *

from evaluation import SceneNetEval


from config_nyuv2 import *

PRUNING_METHODS = ["baseline", "pt", "static", "dynamic"]
NUM_MODELS = 3
PRUNING_RATIOS = [50, 70, 80, 90]
SEG_METRICS = ["mIoU", "Pixel Acc", "err"]
SN_METRICS = ["Angle Mean", "Angle Median", "Angle 11.25", "Angle 22.5", "Angle 30", "Angle 45"]

RESULTS_ROOT = os.path.join(os.environ.get('TMPDIR'), 'results')

baseline_results = {}
model_results = {}

def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def plot_metrics(results, metric_name, save_path, task):
    LINE_STYLES = ['-', '--', '-.', ':']
    plt.figure(figsize=(10, 8))
    colors = {'baseline': 'r', 'pt': 'b', 'static': 'g', 'dynamic': 'c'}  # Add other methods as needed
    baseline_value = None
    
    for method, method_results in results.items():
        color = colors.get(method, 'k')  # Default to 'k' (black) if method not in colors
        for model_num, metrics in method_results.items():
            line_style = LINE_STYLES[model_num % len(LINE_STYLES)]  # Cycle through line styles
            
            if method == "baseline":
              y = metrics[task][metric_name]
              plt.axhline(y=y, color=color, linestyle=line_style, label=f"{method} model {model_num}", linewidth=2)
            else:
                x = PRUNING_RATIOS
                y = [metrics[ratio][task][metric_name] for ratio in PRUNING_RATIOS]
                plt.plot(x, y, marker='o', color=color, linestyle=line_style, label=f"{method} model {model_num}", markersize=10, linewidth=2)

    if baseline_value is not None:
        plt.axhline(y=baseline_value, color=colors['baseline'], linestyle='--', label='baseline', linewidth=2)

    plt.xlabel("Pruning Ratio (%)", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.title(f"Comparison of {metric_name} across models and pruning methods", fontsize=16)
    plt.xticks(PRUNING_RATIOS, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_mean_metrics(results, metric_name, save_path, task):
    plt.figure(figsize=(6, 6))
    colors = {'baseline': 'r', 'pt': 'b', 'static': 'g', 'dynamic': 'c'}  # Add other methods as needed
    mean_metrics = {}
    
    for method, method_results in results.items():
        if method == "baseline":
            baseline_value = np.mean([metrics[task][metric_name] for metrics in method_results.values()])
        else:
            mean_metrics[method] = {}
            for ratio in PRUNING_RATIOS:
                ratio_metrics = [metrics[ratio][task][metric_name] for metrics in method_results.values() if ratio in metrics]
                mean_metrics[method][ratio] = np.mean(ratio_metrics)
    
    for method, metrics in mean_metrics.items():
        color = colors.get(method, 'k')  # Default to 'k' (black) if method not in colors
        x = PRUNING_RATIOS
        y = [metrics[ratio] for ratio in PRUNING_RATIOS]
        plt.plot(x, y, marker='o', color=color, linestyle='-', label=f"{method} mean", markersize=10, linewidth=2)
    
    if 'baseline' in colors:
        plt.axhline(y=baseline_value, color=colors['baseline'], linestyle='-', label=f"baseline model", linewidth=2)

    plt.xlabel("Pruning Ratio (%)", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.title(f"Mean {metric_name} across pruning methods", fontsize=16)
    plt.xticks(PRUNING_RATIOS, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    dataset = 'nyuv2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = NYU_v2(DATA_ROOT, 'test')

    sn_path = os.path.join(RESULTS_ROOT, "sn")
    seg_path = os.path.join(RESULTS_ROOT, "seg")

    os.mkdir(sn_path)
    os.mkdir(seg_path)

    for method in PRUNING_METHODS:
            method_results = {}
            for model_num in range(1, NUM_MODELS+1):

                if method == "baseline":
                    net = SceneNet(TASKS_NUM_CLASS).to(device)
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
                        
                        net = SceneNet(TASKS_NUM_CLASS).to(device)

                        for module in net.modules():
                            # Check if it's basic block
                            if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
                                module = prune.identity(module, 'weight')

                        # need to actually retrieve the model
                        network_name = f"{dataset}_disparse_{method}_{ratio}"

                        path_to_model = os.path.join(os.environ.get('TMPDIR'), "pruned", method, method+str(model_num), "tmp/results", f"best_{network_name}.pth")
                        net.load_state_dict(torch.load(path_to_model))
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

    for metric in SEG_METRICS:
        plot_metrics(model_results, metric, os.path.join(seg_path, metric+'_comparison.png'), 'seg')
        plot_mean_metrics(model_results, metric, os.path.join(seg_path, 'mean_' + metric +'_comparison.png'), 'seg')

    for metric in SN_METRICS:
        plot_metrics(model_results, metric, os.path.join(sn_path, metric+'_comparison.png'), 'sn')
        plot_mean_metrics(model_results, metric, os.path.join(sn_path, 'mean_' + metric +'_comparison.png'), 'sn')


                        
            
    
    