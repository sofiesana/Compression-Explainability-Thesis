import os.path
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import sys
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Get the value of the TMPDIR environment variable
tmpdir = os.environ.get('TMPDIR')

#Importing
quantus_dir = os.path.join(tmpdir, 'quantus/Quantus-Thesis-Version')
sys.path.insert(0, quantus_dir)

from quantus import IROF
from quantus import functions

def plot_all_irof_curves(histories, task, location, class_name = None):
    path = None
    for history in histories:
        history = np.array(history)
        plt.plot(range(len(history)), history, marker='o')
    plt.title('IROF Curve')
    plt.xlabel('Number of Segments Removed')
    plt.grid(True)
    if class_name is not None:
        if task == "sn":
            plt.ylabel('Surface Normal Octant \'' + str(class_name) + '\' Score')
        path = os.path.join(location, "Irof_results/" + task + "/", str(class_name) + ' all_irof.png')
    else:
        plt.ylabel('Class Score')
        path = os.path.join(location, "Irof_results/" + task + "/", 'all_irof.png')
    os.makedirs(path, exist_ok=True)
    plt.savefig(path)
    print("Saved at: ", path)
    plt.close()

def plot_avg_irof_curve(histories, task, location, class_name = None):
    path = None
    # Step 1: Find the length of the longest list
    max_length = max(len(lst) for lst in histories)

    # Step 2: Pad shorter lists with NaN values
    padded_lists = [lst + [np.nan] * (max_length - len(lst)) for lst in histories]

    # Step 3: Convert to a NumPy array
    array = np.array(padded_lists)

    # Step 4: Compute the mean along axis 0, ignoring NaN values
    avg_curve = np.nanmean(array, axis=0)

    plt.plot(range(len(avg_curve)), avg_curve, marker='o')
    plt.title('IROF Curve')
    plt.xlabel('Number of Segments Removed')
    plt.grid(True)

    if class_name is not None:
        if task == "sn":
            plt.ylabel('Surface Normal Octant \'' + str(class_name) + '\' Score')
        path = os.path.join(location, "Irof_results/" + task + "/", str(class_name) + ' avg_irof.png')
    else:
        plt.ylabel('Score')
        path = os.path.join(location, "Irof_results/" + task + "/", 'avg_irof.png')
    os.makedirs(path, exist_ok=True)
    plt.savefig(path)
    print("Saved at: ", path)
    plt.close()

def run_irof_surf_norm(model, model_name, test_loader, location, sem_idx_to_class, device, XPLANATIONS_ROOT, num_images_to_gen_irof = 200):

    class_scores = {}
    class_histories = {}

    for i, gt_batch in enumerate(test_loader):
        if i >= num_images_to_gen_irof:
            break
        print("Image: ", i)
        model.eval()
        gt_batch["img"] = Variable(gt_batch["img"]).to(device)
        if "seg" in gt_batch:
            gt_batch["seg"] = Variable(gt_batch["seg"]).to(device)
        if "depth" in gt_batch:
            gt_batch["depth"] = Variable(gt_batch["depth"]).to(device)
        if "normal" in gt_batch:
            gt_batch["normal"] = Variable(gt_batch["normal"]).to(device)

        print("getting preds")
        preds = model(gt_batch["img"])
        
        img_names = gt_batch["name"]
        image = gt_batch["img"]
        print(img_names)

        # for i in range(len(image)):
        #     img_names[i] = str(img_names[i]).replace('.png', '')

        image_explanations = [] # in form (explanation, norm_num)
        print("getting explanations")
        for norm_num in range(9):
            
            # file_name = "./saved_explanations/" + model_name + "xplanations_new/Surface Normals/X_Image" + str(k) + "_norm" + str(norm_num) + ".npy"

            # NEED TO FIX THIS: ON HABROK YOU WILL HAVE TO UNZIP THE RESULTS FOLDER AND THEN USE THE PATH TO THE EXPLANATIONS
            file_name = os.path.join(XPLANATIONS_ROOT, "xplanations_new/Surface Normals/X_Image" + str(img_names) + "_norm" + str(norm_num) + ".npy")
            if os.path.isfile(file_name):
                loaded_explanation = np.load(file_name)
                image_explanations.append((loaded_explanation, norm_num))
            else:
                print("File not found: ", file_name)

        print("image_explanations: ", image_explanations)


        for explanation, norm_num in image_explanations:
            print("norm_num: ", norm_num)
            irof = IROF(segmentation_method="slic",
                                    perturb_baseline="mean",
                                    perturb_func=functions.perturb_func.baseline_replacement_by_indices,
                                    return_aggregate=False,
                                    display_progressbar=True,
                                    class_category=sem_idx_to_class[norm_num],
                                    num_classes=8, 
                                    task = "sn"
                                    )
            
            x_batch = image
            y_batch = preds
            a_batch = torch.tensor(explanation, dtype=torch.float32).unsqueeze(0)
            
            scores, histories = irof(model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    device=device)
            print(scores)

            if scores is not None:
                if norm_num not in class_scores.keys():
                    class_scores[norm_num] = []
                    class_histories[norm_num] = []

                class_scores[norm_num].extend(scores)
                class_histories[norm_num].extend(histories)
    
    print("Plotting Irof Curves")
    print(class_scores)
    for category in class_scores.keys():
        print("Category: ", category)
        class_name = sem_idx_to_class[category]
        plot_all_irof_curves(class_histories[category], "sn", location, class_name)
        plot_avg_irof_curve(class_histories[category], "sn", location, class_name)

    return class_scores, class_histories  


def irof_caller(model, model_name, test_loader, location, device, XPLANATIONS_ROOT):
    model.eval()
    print("Evaluating Irof for model: ", model_name)

    tasks = ["sn"]

    with torch.no_grad():
        for task in tasks:
            print("Task: ", task)
            if task == "sn":
                sem_idx_to_class = [(-1,-1,-1), (-1, 1, -1), (1, -1, -1), (1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)]
                scores, histories = run_irof_surf_norm(model, model_name, test_loader, location, sem_idx_to_class, device, XPLANATIONS_ROOT, num_images_to_gen_irof=2)

            path = "Irof_results/" + task
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(location, "Irof_results/" + task + "/", model_name + 'histories.pkl'), 'wb') as file:
                pickle.dump(histories, file)
            
            with open(os.path.join(location, "Irof_results/" + task + "/", model_name + 'scores.pkl'), 'wb') as file:
                pickle.dump(scores, file)

    print("Irof Caller Complete")