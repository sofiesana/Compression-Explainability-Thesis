from vojo_irof_hbrk import *
import argparse
from dataloaders import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from irof_utils_hbrk import *
from evaluation import *

# Get the value of the TMPDIR environment variable
tmpdir = os.environ.get('TMPDIR')

NUM_MODELS = 3
PRUNING_RATIOS = [50, 70, 80, 90]

# main
if __name__ == '__main__':   

    torch.cuda.empty_cache()

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

            if method == "baseline":
                location = method + str(model_num)
                XPLANATIONS_ROOT = os.path.join(tmpdir, 'explanations', 'tmp/results/baseline'+str(model_num))


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
                

                torch.cuda.empty_cache()
                net = load_model(device, 'n', task, path_to_model)
                net.eval()

                if task == 'seg':
                    print("Not implemented for semantic segmentation.")
                    break
                elif task == 'sn':
                    print("######### Beginning explanation generation.")
                    irof_caller(net, model_name, test_loader, rslt_path, device, XPLANATIONS_ROOT)
                    
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

                    XPLANATIONS_ROOT = os.path.join(tmpdir, 'explanations', 'tmp/results/'+method+str(model_num), str(ratio))
                    
                    model_name = f"{method} model {model_num} ratio {ratio}"
                    print(model_name)
                    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True)
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
                        irof_caller(net, model_name, test_loader, rslt_path, device, XPLANATIONS_ROOT)
                    else:
                        print("task not recognized")
                        break
                    
                    del net
                    torch.cuda.empty_cache()