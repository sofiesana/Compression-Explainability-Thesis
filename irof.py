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
import matplotlib as plt
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
        task_output = 'nothing'
        for t_id in range(self.num_tasks):
            output = getattr(self, 'task%d_fc1_c0' % (t_id + 1))(feats) + \
                getattr(self, 'task%d_fc1_c1' % (t_id + 1))(feats) + \
                getattr(self, 'task%d_fc1_c2' % (t_id + 1))(feats) + \
                getattr(self, 'task%d_fc1_c3' % (t_id + 1))(feats)
            outputs.append(output)
            # if running explainability experiments, only output task-specific output
            if self.task is not None and task_id == t_id:
                print("task id:", task_id)
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

def get_binary_mask(mask_batch, img_names):
    for i in range(len(mask_batch)):
        name = img_names[i]
        name = str(name).replace('.png', '')
        img = mask_batch[i, :, : None]
        masked_pixels = Image.fromarray(np.repeat(img, 3, axis=-1))
        masked_pixels.save(os.path.join(RESULTS_ROOT, name+'_predicted_mask_'+seg_class+'.jpg'))
        print("masked save location:", RESULTS_ROOT, name+'_predicted_mask_'+seg_class+'.jpg')

def get_gradcam_image(img_names, attributions, image):
    for i in range(len(image)):
        name = img_names[i]
        name = str(name).replace('.png', '')

        og_img = (image[i].cpu().squeeze().permute(1,2,0).numpy())
        og_img = (og_img - og_img.min()) / (og_img.max() - og_img.min())
        
        cam_image = show_cam_on_image(og_img, attributions[i], use_rgb=True)

        cam_image_final = Image.fromarray(cam_image)
        cam_image_final.save(os.path.join(RESULTS_ROOT, name+'_grad_cam_'+seg_class+'.jpg'))
        print("gradcam save location:", RESULTS_ROOT, name+'_grad_cam_'+seg_class+'.jpg')

def get_attributions(model, class_category, class_mask_float, image):
    target_layers = [model.backbone]
    targets = [SemanticSegmentationTarget(class_category, class_mask_float)]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image,
                            targets=targets)
    return grayscale_cam


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IROF Evaluation')
    parser.add_argument(
        '--head', type=str, help='mtl model head: all, ', default="all")
    parser.add_argument(
        '--pruned', type=str, help='is the model pruned?: y, n', default="n")
    parser.add_argument(
        '--task', type=str, help='seg, sn', default="None")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = NYU_v2(DATA_ROOT, 'test')
    test_loader = DataLoader(test_dataset, batch_size=3,
                             num_workers=8, shuffle=True, pin_memory=True)

    pruned = args.pruned
    head = args.head
    task = args.task
    if task == "None":
        task = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, pruned, task)

    all_preds = []
    preds = None
    img_name = None
    image = None

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
        all_preds.append(preds)
        img_names = gt_batch["name"]
        image = gt_batch["img"]

        seg = F.interpolate(preds, (480, 640))
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(CLASS_NAMES)}
        normalized_masks = torch.nn.functional.softmax(preds, dim=1)
        seg_class = 'wall'
        class_category = sem_class_to_idx[seg_class]
        class_mask = normalized_masks.argmax(axis=1).detach().cpu().numpy()
        class_mask_uint8 = 255 * np.uint8(class_mask == class_category)
        class_mask_float = np.float32(class_mask == class_category)

        get_binary_mask(class_mask_uint8, img_names)

        attributions = get_attributions(model, class_category, class_mask_float, image)

        get_gradcam_image(img_names, attributions, image)

        irof = quantus.IROF(segmentation_method="slic",
                                perturb_baseline="mean",
                                perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                                return_aggregate=False,
                                )

        # check this line
        grayscale_cam_batch = np.expand_dims(attributions, axis=0)

        labels = np.unique(gt_batch["seg"].cpu().numpy()[0])
        labels[labels == 255] = 0
        labels = torch.tensor(labels)
        
        scores = irof(model=model,
            x_batch=image,
            y_batch=labels,
            a_batch=grayscale_cam_batch,
            device=device)
        print("scores:", scores)