from quintiles_and_octant_exp_gener_hbrk import *
import os
import torch
import argparse
from dataloaders import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from evaluation import *
from irof_utils_hbrk import *
import time

# Get the value of the TMPDIR environment variable
tmpdir = os.environ.get('TMPDIR')

NUM_MODELS = 3
PRUNING_RATIOS = [50, 70, 80, 90]

def get_sn_image(img_names, preds, location):
    normalized_preds = F.normalize(preds, dim=1)
    resized_preds = F.interpolate(normalized_preds, (480, 640))
    for i, pred in enumerate(resized_preds):
        name = img_names[i]
        sn_output = np.uint8(255*pred.detach().cpu().numpy())
        image_array = np.transpose(sn_output, (1,2,0))
        image = Image.fromarray(image_array)
        path = os.path.join(location, 'poster_images', name+'_pred_sn.jpg')
        image.save(path)

def explanation_generator(test_loader, model, device, task, location, num_images_to_gen_explanation = 1):
    # tasks = ["SemSeg", "Depth", "SurNorm", "multi"]
    # modes = ["Generation", "Evaluation"]
    tasks = [task] # only select the tasks that u wanna generate explantions for
    
    # mode = modes[0]
    surface_norm_classes = [(-1,-1,-1), (-1, 1, -1), (1, -1, -1), (1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)] # signs of x, y and z
    print("Creating Explanations...")
    begin = time.time()

    # evaluating test data
    model.eval()
    road_images_results = [] # List of tuples (perc, cat_class, metric, input_number)
    # torch.manual_seed(6277)

    with torch.no_grad():  # operations inside don't track history  (e.g. loss calculation) 
        for i, gt_batch in enumerate(test_loader):
            if i > num_images_to_gen_explanation:
                print("Finished generating explanations for ", num_images_to_gen_explanation, " images.")
                break
            batch_start = time.time()
            model.eval()
            gt_batch["img"] = Variable(gt_batch["img"]).to(device)
            if "seg" in gt_batch:
                gt_batch["seg"] = Variable(gt_batch["seg"]).to(device)
            if "depth" in gt_batch:
                gt_batch["depth"] = Variable(gt_batch["depth"]).to(device)
            if "normal" in gt_batch:
                gt_batch["normal"] = Variable(gt_batch["normal"]).to(device)

            norm_label = gt_batch["normal"]
            # norm_label = norm_label.float() / 255.0
            # print("norm_label:", norm_label)
            # norm_label = (norm_label + 1) / 2

            print("getting preds")
            preds = model(gt_batch["img"])
            
            
            img_names = gt_batch["name"]
            image = gt_batch["img"]
            
            get_sn_image(img_names, preds, None, location)

            start_task = time.time()
            print("Time till start of task: ", round(start_task - begin, 3), " seconds")
            """ Preprocess the regresion outputs """
            image_np_depth_quin, norms_one_hot = None, None
            if task == "SurNorm": # Get octants
                norms_one_hot = preprocess_surface_normals(preds)
            # break
                
            pre_pro_time = time.time()
            print("Time to pre-processing: ", round(pre_pro_time - start_task, 3), " seconds")

            """ Generate Explanations """
            explanations_generated = generate_explanations(task, model, image, preds, image_np_depth_quin, norms_one_hot, img_names, device, location)

            batch_end = time.time()
            print("Time for one batch: ", round(batch_end - batch_start, 3), " seconds")



# main
if __name__ == '__main__':   

    # torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='IROF Evaluation')
    parser.add_argument(
        '--method', type=str, help='pruning method to evaluate: baseline, pt, dynamic, static', default="baseline")
    parser.add_argument(
        '--task', type=str, help='seg, sn', default="None")
    parser.add_argument(
        '--model_num', type=int, help='1,2,3', default=None)
    args = parser.parse_args()
    print("parsing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading data")
    test_dataset = NYU_v2(DATA_ROOT, 'test')
    print("data loaded")

    method = args.method
    task = args.task
    model_number = args.model_num
    if task == "None":
        task = None        

    if model_number is None:
        print("No model number given, running on all models")
        model_num_list = [1,2,3]
    else:
        model_num_list = [model_number]
    
    PRUNING_METHODS = [method]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    preds = None
    img_name = None
    image = None
    dataset = 'nyuv2'
    all_scores = {}
    all_histories = {}

    for method in PRUNING_METHODS:
        print("check 1")
        for model_num in model_num_list:
            location = method + str(model_num)

            if method == "baseline":

                rslt_path = os.path.join(RESULTS_ROOT, location)
                if not os.path.isdir(rslt_path):
                    os.makedirs(rslt_path)
                
                print("baseline model " + str(model_num))
                test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=True, pin_memory=True)
                evaluator = SceneNetEval(
                        device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
                                    
                network_name = f"{dataset}_{method}"
                path_to_model = os.path.join(os.environ.get('TMPDIR'), method, method + str(model_num), "tmp/results", f"best_{network_name}.pth")
                

                torch.cuda.empty_cache()
                net = load_model(device, 'n', task, path_to_model)
                net.eval()

                if task == 'seg':
                    print("Not implemented for semantic segmentation.")
                    break
                elif task == 'sn':
                    print("######### Beginning explanation generation.")
                    explanation_generator(test_loader, net, device, "SurNorm", num_images_to_gen_explanation=200, location = rslt_path)
                    
                else:
                    print("task not recognized")
                    break

            else:
                
                for ratio in PRUNING_RATIOS:
                    print("check 2")

                    # print(torch.cuda.memory_summary())
                    location = os.path.join(method + str(model_num), str(ratio))

                    rslt_path = os.path.join(RESULTS_ROOT, location)
                    if not os.path.isdir(rslt_path):
                        os.makedirs(rslt_path)
                    
                    print(f"{method} model {model_num} ratio {ratio}")
                    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=True, pin_memory=True)
                    evaluator = SceneNetEval(
                            device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)

                    network_name = f"{dataset}_disparse_{method}_{ratio}"

                    path_to_model = os.path.join(os.environ.get('TMPDIR'), "pruned", method, method+str(model_num), "tmp/results", f"best_{network_name}.pth")

                    torch.cuda.empty_cache()
                    net = load_model(device, 'y', task, path_to_model)
                    net.eval()
                    print("model loaded")

                    if task == 'seg':
                        print("Not implemented for semantic segmentation.")
                        break
                    elif task == 'sn':
                        print("######### Beginning explanation generation.")
                        explanation_generator(test_loader, net, device, "SurNorm", num_images_to_gen_explanation=200, location = rslt_path)
                    else:
                        print("task not recognized")
                        break
                    
                    del net
                    torch.cuda.empty_cache()