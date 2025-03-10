from vojo_irof import *
import argparse
from dataloaders import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from irof_utils import *
from evaluation import *

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

    DATA_ROOT = 'data/nyu_v2_with_val/new_data/nyu_v2'

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
    PRUNING_RATIOS = [90]
    RESULTS_ROOT = 'results'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

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
                
                model_name = "baseline model " + str(model_num)
                print(model_name)
                
                test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)
                evaluator = SceneNetEval(
                        device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)
                                    
                network_name = f"{dataset}_{method}"
                path_to_model = os.path.join(os.environ.get('TMPDIR'), method, method + str(model_num), "tmp/results", f"best_{network_name}.pth")
                

                # torch.cuda.empty_cache()
                net = load_model(device, 'n', task, path_to_model)
                net.eval()

                if task == 'seg':
                    print("Not implemented for semantic segmentation.")
                    break
                elif task == 'sn':
                    print("######### Beginning irof evaluation")
                    
                else:
                    print("task not recognized")
                    break

            else:
                
                for ratio in PRUNING_RATIOS:
                    print("check 2")

                    # print(torch.cuda.memory_summary())
                    location = os.path.join(method + str(model_num), str(ratio))
                    location = "results/dynamic3/90"

                    rslt_path = os.path.join(RESULTS_ROOT, location)
                    if not os.path.isdir(rslt_path):
                        os.makedirs(rslt_path)
                    
                    model_name = f"{method} model {model_num} ratio {ratio}"
                    print(model_name)
                    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)
                    evaluator = SceneNetEval(
                            device, TASKS, TASKS_NUM_CLASS, IMAGE_SHAPE, dataset, DATA_ROOT)

                    network_name = f"{dataset}_disparse_{method}_{ratio}"

                    # path_to_model = os.path.join(os.environ.get('TMPDIR'), "pruned", method, method+str(model_num), "tmp/results", f"best_{network_name}.pth")
                    path_to_model = "models/best_nyuv2_disparse_dynamic_90.pth"
                    # torch.cuda.empty_cache()
                    net = load_model(device, 'y', task, path_to_model)
                    net.eval()
                    print("model loaded")

                    if task == 'seg':
                        print("Not implemented for semantic segmentation.")
                        break
                    elif task == 'sn':
                        print("######### Beginning irof.")
                        irof_caller(net, model_name, test_loader, location, device)
                    else:
                        print("task not recognized")
                        break
                    
                    del net
                    # torch.cuda.empty_cache()