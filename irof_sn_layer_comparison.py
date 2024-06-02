from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
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
        print("task:", self.task)

        # if running explainability experiments
        if self.task is not None:
            if self.task == 'seg':
                task_id = 0
            if self.task == 'sn':
                task_id = 1
        print("task:", task_id)
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats) + \
                getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats) + \
                getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats) + \
                getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats)
            outputs.append(output)
            # if running explainability experiments, only output task-specific output
            if self.task is not None and task_id == t_id:
                return output
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

class EntireOutputTarget:
    def __call__(self, model_output):
        return model_output.sum()

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

def get_gradcam_image_sn(img_names, attributions, image, layer_name):
    for i, name in enumerate(img_names):
        og_img = (image[i].cpu().squeeze().permute(1, 2, 0).numpy())
        og_img = (og_img - og_img.min()) / (og_img.max() - og_img.min())
        
        cam_image = show_cam_on_image(og_img, attributions[i], use_rgb=True)

        cam_image_final = Image.fromarray(cam_image)
        path = os.path.join(RESULTS_ROOT, f"{name}_grad_cam_{layer_name}.jpg")
        cam_image_final.save(path)

def get_sn_image(img_names, preds):
    normalized_preds = F.normalize(preds, dim=1)
    resized_preds = F.interpolate(normalized_preds, (480, 640))
    for i, pred in enumerate(resized_preds):
        name = img_names[i]
        sn_output = np.uint8(255*pred.detach().cpu().numpy())
        image_array = np.transpose(sn_output, (1,2,0))
        image = Image.fromarray(image_array)
        path = os.path.join(RESULTS_ROOT, name+'_pred_sn.jpg')
        image.save(path)

def get_sn_attributions(model, image, layer):
    target_layers = [layer]
    targets = [EntireOutputTarget() for _ in range(len(image))]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image, targets=targets)
    return grayscale_cam

def plot_all_irof_curves_sn(histories, layer_name):
    for history in histories:
        history = np.array(history)
        plt.plot(range(len(history)), history, marker='o')
    plt.title('AOC Curve')
    plt.xlabel('Number of Segments Removed')
    plt.ylabel('Class Score')
    plt.grid(True)
    path = os.path.join(RESULTS_ROOT, f"all_irof_{layer_name}.jpg")
    plt.savefig(path)
    plt.close()

def plot_avg_irof_curve_sn(histories, layer_name):
    # Step 1: Find the length of the longest list
    max_length = max(len(lst) for lst in histories)

    # Step 2: Pad shorter lists with NaN values
    padded_lists = [lst + [np.nan] * (max_length - len(lst)) for lst in histories]

    # Step 3: Convert to a NumPy array
    array = np.array(padded_lists)

    # Step 4: Compute the mean along axis 0, ignoring NaN values
    avg_curve = np.nanmean(array, axis=0)

    plt.plot(range(len(avg_curve)), avg_curve, marker='o')
    plt.title('AOC Curve')
    plt.xlabel('Number of Segments Removed')
    plt.ylabel('Score')
    plt.grid(True)
    path = os.path.join(RESULTS_ROOT, f"avg_irof_{layer_name}.jpg")
    plt.savefig(path)
    plt.close()

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

    preds = None
    img_name = None
    image = None
    all_scores = []
    all_histories = []

    layers_to_try = [model.backbone, model.backbone.blocks[3][2].relu, model.backbone.blocks[3][2].conv2]  # Example layers
    layer_names = ["backbone", "blocks[3][2].relu", "blocks[3][2].conv2"]

    for layer, layer_name in zip(layers_to_try, layer_names):
        os.mkdir(os.path.join(RESULTS_ROOT, layer_name))
        test_loader = DataLoader(test_dataset, batch_size=10,
                             num_workers=8, shuffle=False, pin_memory=True)
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
            print(preds)
            print(F.normalize(preds, dim=1))
            
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

            attributions = get_sn_attributions(model, image, layer)
            get_gradcam_image_sn(img_names, attributions, image, layer_name)
            get_sn_image(img_names, preds)

            scores, histories = irof(model=model,
                    x_batch=gt_batch["img"],
                    y_batch=preds,
                    a_batch=attributions,
                    device=device)
            
            if scores is not None:
                all_scores.extend(scores)
                all_histories.extend(histories)
            break
        
        print("mean score:", np.mean(np.array(scores)))

        plot_all_irof_curves_sn(all_histories, layer_name)
        plot_avg_irof_curve_sn(all_histories, layer_name)

        with open(os.path.join(RESULTS_ROOT, layer_name, 'histories.pkl'), 'wb') as file:
            pickle.dump(all_histories, file)
        
        with open(os.path.join(RESULTS_ROOT, layer_name, 'scores.pkl'), 'wb') as file:
            pickle.dump(all_scores, file)