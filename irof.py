import matplotlib as plt
import pickle
import tqdm
from prune_utils import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import quantus
from quantus.metrics.faithfulness.irof import IROF
from evaluation import *
from scene_net import *
from dataloaders import *
import argparse

# Get the value of the TMPDIR environment variable
tmpdir = os.environ.get('TMPDIR')

# Define the directory path you want to access
nyuv2_dir = os.path.join(tmpdir, 'nyuv2/new_data/nyu_v2/')

# Get the value of the TMPDIR environment variable
tmpdir = os.environ.get('TMPDIR')
# Define the directory path you want to access
pt_dir = os.path.join(tmpdir, 'pt/tmp/results/best_nyuv2_baseline.pth')


RESULTS_ROOT = '/data/preds'
DATA_ROOT = nyuv2_dir
MODEL_ROOT = pt_dir
TASKS = ["seg", "sn"]
TASKS_NUM_CLASS = [40, 3]
IMAGE_SHAPE = (480, 640)


def load_model(device, pruned):
    model = SceneNet(TASKS_NUM_CLASS).to(device)

    if pruned == 'y':
        for module in model.modules():
            # Check if it's basic block
            if isinstance(module, nn.modules.conv.Conv2d) or isinstance(module, nn.modules.Linear):
                module = prune.identity(module, 'weight')

    # Load the state dictionary into your model
    model.load_state_dict(torch.load(MODEL_ROOT))
    model.cuda()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IROF Evaluation')
    parser.add_argument(
        '--head', type=str, help='mtl model head: all, ', default="all")
    parser.add_argument(
        '--pruned', type=str, help='is the model pruned?: y, n', default="n")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = NYU_v2(DATA_ROOT, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1,
                             num_workers=8, shuffle=True, pin_memory=True)

    pruned = args.pruned
    head = args.head

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(device, pruned)

    irof_metric = quantus.IROF(
        segmentation_method='slic', abs=False, normalise=True)

    print(model)

    all_preds = []

    # Iterate over batches in the dataloader
    for i, gt_batch in enumerate(test_loader):
        model.eval()
        gt_batch["img"] = Variable(gt_batch["img"]).to(device)
        if "seg" in gt_batch:
            gt_batch["seg"] = Variable(gt_batch["seg"]).to(device)
        if "depth" in gt_batch:
            gt_batch["depth"] = Variable(gt_batch["depth"]).to(device)
        if "normal" in gt_batch:
            gt_batch["normal"] = Variable(gt_batch["normal"]).to(device)

        # irof = quantus.IROF(segmentation_method="slic",
        #                     perturb_baseline="mean",
        #                     perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        #                     return_aggregate=False,
        #                     )

        preds = model(gt_batch["img"])
        all_preds.append(preds)

        if i == 0:
            print(type(preds))
            print("preds:", preds)
            print("seg:", preds[0].shape)
            print("sn:", preds[1].shape)

        # self.seg_pred = preds[self.tasks.index('seg')]
        # self.seg_output = F.interpolate(self.seg_pred, size=IMAGE_SHAPE)

        # # a_batch needs to be calculated

        # scores = {method: irof(model=model,
        # x_batch=gt_batch["img"],
        # y_batch=gt_batch["seg"],
        # a_batch=None,
        # device=device,
        # explain_func=quantus.explain,
        # explain_func_kwargs={"method": method}) for method in ["Saliency"]}

        # print(scores.shape)

    # Open a file in binary write mode
    with open(RESULTS_ROOT+'/predictions.pkl', 'wb') as f:
        # Pickle the object into the file
        pickle.dump(all_preds, f)

        # print(scores)
        # fig, ax = irof.plot(results=scores)

        # # Save the plot to a file
        # fig.savefig('pixel_flipping_plot.png')
        # plt.close(fig)  # Close the figure to free up memory
