from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from dataloaders import *
from scene_net import *
from evaluation import *
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

nyuv2_dir = os.path.join(tmpdir, 'nyuv2/new_data/nyu_v2/')
pt_dir = os.path.join(tmpdir, 'pt/tmp/results/best_nyuv2_baseline.pth')
results_dir = os.path.join(tmpdir, 'results')

RESULTS_ROOT = results_dir
DATA_ROOT = nyuv2_dir
MODEL_ROOT = pt_dir

TASKS = ["seg", "sn"]
TASKS_NUM_CLASS = [40, 3]
IMAGE_SHAPE = (480, 640)
CLASS_NAMES = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes',
    'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']


class SceneNet(nn.Module):
    def __init__(self, num_classes_tasks, task=None, skip_layer=0):
        super().__init__()
        block = BasicBlock
        layers = [3, 4, 6, 3]
        self.backbone = Deeplab_ResNet_Backbone(block, layers)
        self.num_tasks = len(num_classes_tasks)
        self.task = task

        for t_id, num_class in enumerate(num_classes_tasks):
            setattr(self, 'task%d_fc1_c0' % (
                t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=6))
            setattr(self, 'task%d_fc1_c1' % (
                t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=12))
            setattr(self, 'task%d_fc1_c2' % (
                t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=18))
            setattr(self, 'task%d_fc1_c3' % (
                t_id + 1), Classification_Module(512 * block.expansion, num_class, rate=24))

        self.layers = layers
        self.skip_layer = skip_layer

        self.policys = []
        for t_id in range(self.num_tasks):
            self.policys.append(None)
    ################################################################################################

    def arch_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'logits' in name:
                params.append(param)
        return params

    def backbone_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'backbone' in name:
                params.append(param)
        return params

    def task_specific_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'task' in name and 'fc' in name:
                params.append(param)
        return params

    def network_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if not ('task' in name and 'logits' in name):
                params.append(param)
        return params
    ################################################################################################

    def forward(self, img, num_train_layers=None, hard_sampling=False, mode='train'):
        if num_train_layers is None:
            num_train_layers = sum(self.layers) - self.skip_layer

        num_train_layers = min(
            sum(self.layers) - self.skip_layer, num_train_layers)
        # Generate features
        cuda_device = img.get_device()
        # Use a shared trunk structure. Shared extracted features
        outputs = []
        feats = self.backbone(img)
        task_id = 0

        # if running explainability experiments
        if self.task is not None:
            if self.task == 'seg':
                task_id = 0
            if self.task == 'sn':
                task_id = 1
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats) + \
                getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats) + \
                getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats) + \
                getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats)
            outputs.append(output)
            # if running explainability experiments, only output task-specific output
            if self.task is not None and task_id == t_id:
                outputs = output
                break
        return outputs
################################################################################################

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

def load_model(device, pruned, task):
    model = SceneNet(TASKS_NUM_CLASS, task=task).to(device)

    if pruned == 'y':
        for module in model.modules():
            # Check if it's basic block
            if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
                module = prune.identity(module, 'weight')

    # Load the state dictionary into your model
    model.load_state_dict(torch.load(MODEL_ROOT))
    model.cuda()

    return model

# Helper function to generate all combinations of x, y, z
def generate_xyz_combinations():
    coordinates = [-1, 1]
    combinations = list(itertools.product(coordinates, repeat=3))
    return {idx: comb for idx, comb in enumerate(combinations)}

def get_resized_binary_mask(img_names, preds, category, category_index, task):
    resized_preds = F.interpolate(preds, (480, 640))
    if task == "seg":
        normalized_masks = torch.nn.functional.softmax(resized_preds, dim=1)
        class_mask = normalized_masks.argmax(axis=1).detach().cpu().numpy()
        class_mask_uint8 = 255 * np.uint8(class_mask == category_index)
    elif task == "sn":
        x_pred, y_pred, z_pred = resized_preds[:, 0, :, :], resized_preds[:, 1, :, :], resized_preds[:, 2, :, :]
        combined_mask = (x_pred == category[0]) & (y_pred == category[1]) & (z_pred == category[2])
        class_mask_uint8 = 255 * np.uint8(combined_mask.detach().cpu().numpy())
    
    for i in range(len(class_mask_uint8)):
        name = img_names[i]
        img = class_mask_uint8[i, :, :, None]
        masked_pixels = Image.fromarray(np.repeat(img, 3, axis=-1))
        path = os.path.join(RESULTS_ROOT, category, name + '_predicted_mask.jpg')
        masked_pixels.save(path)
        print("masked save location:", path)

def get_gradcam_image(img_names, attributions, image, category):
    for i in range(len(image)):
        name = img_names[i]
        og_img = (image[i].cpu().squeeze().permute(1,2,0).numpy())
        og_img = (og_img - og_img.min()) / (og_img.max() - og_img.min())
        cam_image = show_cam_on_image(og_img, attributions[i], use_rgb=True)
        cam_image_final = Image.fromarray(cam_image)
        path = os.path.join(RESULTS_ROOT, category, name + '_grad_cam.jpg')
        cam_image_final.save(path)
        print("gradcam save location:", path)

def get_attributions(model, category, class_mask_float, image, task):
    target_layers = [model.backbone]
    if task == "seg":
        targets = [SemanticSegmentationTarget(category, class_mask) for class_mask in class_mask_float]
    elif task == "sn":
        targets = [SemanticSegmentationTarget(comb, class_mask) for class_mask in class_mask_float]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image, targets=targets)
    return grayscale_cam

def get_binary_mask(preds, category, task):
    if task == "seg":
        normalized_masks = torch.nn.functional.softmax(preds, dim=1)        
        class_mask = normalized_masks.argmax(axis=1).detach().cpu().numpy()
        class_mask_float = np.float32(class_mask == category)
    elif task == "sn":
        resized_preds = F.interpolate(preds, (480, 640))
        x_pred, y_pred, z_pred = resized_preds[:, 0, :, :], resized_preds[:, 1, :, :], resized_preds[:, 2, :, :]
        combined_mask = (x_pred == category[0]) & (y_pred == category[1]) & (z_pred == category[2])
        class_mask_float = np.float32(combined_mask.detach().cpu().numpy())
    return class_mask_float

def plot_all_irof_curves(histories, class_name):
    for history in histories:
        history = np.array(history)
        plt.plot(range(len(history)), history, marker='o')
    plt.title('AOC Curve')
    plt.xlabel('Number of Segments Removed')
    plt.ylabel('Class ' + class_name + ' Score')
    plt.grid(True)
    path = os.path.join(RESULTS_ROOT, class_name, 'all_irof.png')
    plt.savefig(path)
    plt.close()

def plot_avg_irof_curve(histories, class_name):
    max_length = max(len(lst) for lst in histories)
    padded_lists = [lst + [np.nan] * (max_length - len(lst)) for lst in histories]
    array = np.array(padded_lists)
    avg_curve = np.nanmean(array, axis=0)
    plt.plot(range(len(avg_curve)), avg_curve, marker='o')
    plt.title('AOC Curve')
    plt.xlabel('Number of Segments Removed')
    plt.ylabel('Class ' + class_name + ' Score')
    plt.grid(True)
    path = os.path.join(RESULTS_ROOT, class_name, 'avg_irof.png')
    plt.savefig(path)
    plt.close()

def make_class_directories(sem_idx_to_cls, class_indices):
    for i in range(class_indices):
        class_name = sem_idx_to_cls[i]        
        path = os.path.join(RESULTS_ROOT, class_name)
        os.mkdir(path)
        print("Directory '% s' created" % path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IROF Evaluation')
    parser.add_argument('--head', type=str, help='mtl model head: all, ', default="all")
    parser.add_argument('--pruned', type=str, help='is the model pruned?: y, n', default="n")
    parser.add_argument('--task', type=str, help='seg, sn', default="None")
    parser.add_argument('--irof', type=str, help='mean, uniform', default="mean")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = NYU_v2(DATA_ROOT, 'test')
    test_loader = DataLoader(test_dataset, batch_size=10, num_workers=8, shuffle=True, pin_memory=True)

    pruned = args.pruned
    head = args.head
    task = args.task
    irof_version = args.irof
    if task == "None":
        task = None

    print("task:", task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, pruned, task)

    all_preds = []
    preds = None
    img_name = None
    image = None
    class_scores = {}
    class_histories = {}

    if task == "seg":
        categories = {idx: cls for (idx, cls) in enumerate(CLASS_NAMES)}
    elif task == "sn":
        categories = generate_xyz_combinations()

    for i, gt_batch in enumerate(test_loader):
        if i == 2:
            break
        
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
        
        for category_index, category in categories.items():

            category_name = category if task == "seg" else f"comb_{category_index}"
            print(category_name)

            irof = quantus.IROF(segmentation_method="slic",
                                perturb_baseline=irof_version,
                                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                return_aggregate=False,
                                class_category=category,
                                class_name=category_name,
                                num_classes=len(categories),
                                task=task
                                )
            
            valid_indices = []

            class_mask_float = get_binary_mask(preds, category_name, task)
            attributions = get_attributions(model, category_name, class_mask_float, image, task)


            for i, img_seg in enumerate(torch.argmax(preds, axis=1)):
                if task == "seg" and category_index not in img_seg:
                    print(category_name, " not in image ", str(img_names[i]))
                if task == "sn" and category not in img_seg:
                    print(category_name, " not in image ", str(img_names[i]))
                if np.all((attributions[i] == 0)):
                    print("attributions all zero for image ", str(img_names[i]))
                if ((task == "seg" and category_index in img_seg) or (task == "sn" and category in img_seg)) and not np.all((attributions[i] == 0)):
                    valid_indices.append(i)

            print("valid:", valid_indices)

            if valid_indices:
                path = os.path.join(RESULTS_ROOT, category_name)
                if not os.path.isdir(path):
                    os.mkdir(path)
                print("Directory '% s' created" % path)

                y_batch = preds.argmax(axis=1)
                
                reduced_image_names = np.array(img_names)[valid_indices]
                y_batch = y_batch[valid_indices]
                x_batch = image[valid_indices]
                a_batch = attributions[valid_indices]
                reduced_preds = preds[valid_indices]

                get_resized_binary_mask(reduced_image_names, reduced_preds, category_name, category_index, task)

                class_mask_float = get_binary_mask(reduced_preds, category_name, task)
                attributions = get_attributions(model, category_name, class_mask_float, x_batch, task)

                get_gradcam_image(reduced_image_names, a_batch, x_batch, category_name)

                scores, histories = irof(model=model,
                                         x_batch=x_batch,
                                         y_batch=y_batch,
                                         a_batch=a_batch,
                                         true_batch=gt_batch["normal"],
                                         device=device)

                if scores is not None:
                    if category_name not in class_scores.keys():
                        class_scores[category_name] = []
                        class_histories[category_name] = []

                    class_scores[category_name].extend(scores)
                    class_histories[category_name].extend(histories)

    print(class_scores)

    mean_aoc = {}

    for category in class_scores.keys():
        category_name = category if task == "seg" else f"comb_{category}"
        mean_aoc[category_name] = np.mean(np.array(class_scores[category_name]))
        plot_all_irof_curves(class_histories[category_name], category_name)
        plot_avg_irof_curve(class_histories[category_name], category_name)

    print(mean_aoc)

    with open(os.path.join(RESULTS_ROOT, 'histories.pkl'), 'wb') as file:
        pickle.dump(class_histories, file)
    
    with open(os.path.join(RESULTS_ROOT,'/scores.pkl'), 'wb') as file:
        pickle.dump(class_scores, file)