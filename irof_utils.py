import os 
from PIL import Image
from prune_utils import *
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import matplotlib.pyplot as plt

tmpdir = os.environ.get('TMPDIR')
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

from scene_net import *

class SceneNetIROF(nn.Module):
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
    
class EntireOutputTarget:
    def __call__(self, model_output):
        return model_output.sum()

def load_model(device, pruned, task, model_path):
    model = SceneNet(TASKS_NUM_CLASS, task=task).to(device)

    if pruned == 'y':
        for module in model.modules():
            # Check if it's basic block
            if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
                module = prune.identity(module, 'weight')

    # Load the state dictionary into your model
    model.load_state_dict(torch.load(model_path))
    model.cuda()

    return model

def get_resized_binary_mask(img_names, preds, class_name, class_category, location):
    resized_preds = F.interpolate(preds, (480, 640))
    normalized_masks = torch.nn.functional.softmax(resized_preds, dim=1)
    class_mask = normalized_masks.argmax(axis=1).detach().cpu().numpy()
    class_mask_uint8 = 255 * np.uint8(class_mask == class_category)

    for i in range(len(class_mask_uint8)):
        name = img_names[i]
        img = class_mask_uint8[i, :, :, None]
        masked_pixels = Image.fromarray(np.repeat(img, 3, axis=-1))

        path = os.path.join(RESULTS_ROOT, location, class_name, name+'_predicted_mask.jpg')
        masked_pixels.save(path)

def get_sn_image(img_names, preds, location):
    ### requires normalized predictions
    for i, pred in enumerate(preds):
        name = img_names[i]
        sn = F.interpolate(pred, (480, 640))
        sn_output = np.uint8(255*sn.detach().cpu().numpy())
        image_array = np.transpose(sn_output[2], (1, 2, 0))
        image = Image.fromarray(image_array)
        path = os.path.join(RESULTS_ROOT, location, name+'_pred_sn.jpg')
        image.save(path)

def get_gradcam_image(img_names, attributions, image, location, class_name = 'None'):
    for i in range(len(image)):
        name = img_names[i]
        og_img = (image[i].cpu().squeeze().permute(1,2,0).numpy())
        og_img = (og_img - og_img.min()) / (og_img.max() - og_img.min())
        
        cam_image = show_cam_on_image(og_img, attributions[i], use_rgb=True)

        cam_image_final = Image.fromarray(cam_image)
        if class_name is not None:
            path = os.path.join(RESULTS_ROOT, location, class_name, name+'_grad_cam.jpg')
        else:
            path = os.path.join(RESULTS_ROOT, location, name+'_grad_cam.jpg')
        cam_image_final.save(path)

def get_attributions(model, image, class_category = None, class_mask_float = None):
    target_layers = [model.backbone]
    if class_category is not None:
        targets = [SemanticSegmentationTarget(class_category, class_mask) for class_mask in class_mask_float]
    else:
        targets = [EntireOutputTarget()]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image,
                            targets=targets)
    return grayscale_cam

def get_binary_mask(preds, class_category):
    normalized_masks = torch.nn.functional.softmax(preds, dim=1)        
    class_mask = normalized_masks.argmax(axis=1).detach().cpu().numpy()
    class_mask_float = np.float32(class_mask == class_category)
    return class_mask_float

def plot_all_irof_curves(histories, location, class_name = None):
    for history in histories:
        history = np.array(history)
        plt.plot(range(len(history)), history, marker='o')
    plt.title('IROF Curve')
    plt.xlabel('Number of Segments Removed')
    plt.grid(True)
    if class_name is not None:
        plt.ylabel('Class \'' + class_name + '\' Score')
        path = os.path.join(RESULTS_ROOT, location, class_name, 'all_irof.png')
    else:
        plt.ylabel('Class Score')
        path = os.path.join(RESULTS_ROOT, location, 'all_irof.png')
    plt.savefig(path)
    plt.close()

def plot_avg_irof_curve(histories, location, class_name = None):
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
        plt.ylabel('Class \'' + class_name + '\' Score')
        path = os.path.join(RESULTS_ROOT, location, class_name, 'avg_irof.png')
    else:
        plt.ylabel('Score')
        path = os.path.join(RESULTS_ROOT, 'avg_irof.png')
    path = os.path.join(RESULTS_ROOT, location, class_name, 'avg_irof.png')
    plt.savefig(path)
    plt.close()

def make_class_directories(sem_idx_to_cls, class_indices):
    for i in range(class_indices):
        class_name = sem_idx_to_cls[i]        
        # Parent Directory path 
        path = os.path.join(RESULTS_ROOT, class_name)
        os.mkdir(path)
        print("Directory '% s' created" % path)