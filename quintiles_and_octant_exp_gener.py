import time 
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from irof_utils import *
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

""" Explanation Generation Utility Functions """

class SemanticSegmentationTarget:
    def __init__(self, category, masks, task_type="semantic"):
        self.category = category
        self.task_type = task_type
        if task_type == "multi":
            self.mask = [torch.from_numpy(mask).requires_grad_(True) for mask in masks]
        else: 
            self.mask = torch.from_numpy(masks).requires_grad_(True)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        if self.task_type == "semantic":
            outputs = model_output
            output = outputs[0]
            output = output.squeeze(0)
            if self.category == "all":
                return (output[:, :, :] * self.mask).sum()
            return (output[self.category, :, :] * self.mask).sum()
        elif self.task_type == "depth":
            outputs = model_output
            output = outputs[1]
            return (output.squeeze(0) * self.mask).sum()
        elif self.task_type == "normals":
            return (model_output.squeeze(0) * self.mask).sum()
        elif self.task_type == "multi":
            sums = 0
            for i in range(3):
                sums += (model_output[i].squeeze(0) * self.mask[i]).sum()
            return sums
    
    def __str__(self):
        return f"Category: {self.category}, Mask shape: {self.mask.shape}"

def show_mask(test_data, test_pred, cat_class):
    normalized_mask = torch.nn.functional.softmax(test_pred, dim=1).cpu()
    mask_one = normalized_mask[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    mask_one_uint8 = 255 * np.uint8(mask_one == cat_class)
    mask_one_float = np.float32(mask_one == cat_class)            

    repeated_mask = np.repeat(mask_one_uint8[:, :, None], 3, axis=-1)

    return mask_one_float, repeated_mask

def show_mask_depth_and_norms(test_data, test_pred, cat_class):
    mask_one = test_pred[:, :, cat_class]
    mask_one_uint8 = 255 * np.uint8(mask_one == 1)
    mask_one_float = np.float32(mask_one == 1)      

    repeated_mask = np.repeat(mask_one_uint8[:, :, None], 3, axis=-1)

    return mask_one_float, repeated_mask

def show_seg_grad_cam(multi_task_model, test_data, cat_class, mask_one_float, device, k, task_type="semantic"):
    if task_type == "normals":
        layer = [multi_task_model.backbone.blocks[3][2].conv2]
    target_layers = layer
    targets = [SemanticSegmentationTarget(cat_class, mask_one_float, task_type)]
    test_data.requires_grad = True

    og_img = (test_data[0].cpu().squeeze().permute(1,2,0).numpy())
    og_img = (og_img - og_img.min()) / (og_img.max() - og_img.min())
    

    with torch.enable_grad():
        with GradCAM(model=multi_task_model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=test_data.to(device), targets=targets)[0, :]
            cam_image = show_cam_on_image(og_img, grayscale_cam, use_rgb=True)
            cam_image_final = Image.fromarray(cam_image)
            cam_image_final.save(f'poster_images/{k}_cam_image_{cat_class}.png')

    return cam_image, grayscale_cam


""" Preproccesing Functions """

def assign_octile(x, y, z, norms_one_hot, norms_octiles, i, j):
    if z < 0:
        if x < 0 and y < 0:
            norms_one_hot[i][j][0] = 1
            norms_octiles[i][j] = 0
        if x < 0 and y >= 0:
            norms_one_hot[i][j][1] = 1
            norms_octiles[i][j] = 1
        if x >= 0 and y < 0:
            norms_one_hot[i][j][2] = 1
            norms_octiles[i][j] = 2
        if x >= 0 and y >= 0:
            norms_one_hot[i][j][3] = 1
            norms_octiles[i][j] = 3
    if z >= 0:
        if x < 0 and y < 0:
            norms_one_hot[i][j][4] = 1
            norms_octiles[i][j] = 4
        if x < 0 and y >= 0:
            norms_one_hot[i][j][5] = 1
            norms_octiles[i][j] = 5
        if x >= 0 and y < 0:
            norms_one_hot[i][j][6] = 1
            norms_octiles[i][j] = 6
        if x >= 0 and y >= 0:
            norms_one_hot[i][j][7] = 1
            norms_octiles[i][j] = 7

    return norms_one_hot, norms_octiles

def preprocess_surface_normals(test_pred_full):
    # test_pred_full = test_pred_full.float() / 255.0
    # test_pred_full = test_pred_full * 2 - 1

    batch_size, channels, n, m = test_pred_full.shape
    test_pred_full = F.normalize(test_pred_full, p=2, dim=1)
    test_pred_full = test_pred_full * 2 - 1
    # image_np_normal = F.normalize(test_pred_full, p=2, dim=2)
    image_np_normal = test_pred_full.cpu().numpy().squeeze(0).transpose(1, 2, 0)
    
    h, w, d = image_np_normal.shape
    norms_one_hot = np.zeros((h, w, 8))
    norms_octiles = np.zeros((h, w))
    coord_product = np.zeros((h, w))
    for i in range(len(image_np_normal)):
        for j in range(len(image_np_normal[i])):
            x, y, z = image_np_normal[i][j]
            # if i <= 90 and j <= 90 and i >= 70 and j >= 70:
            #     print(i, j, " --  ", x, y, z, " --  ", image_np_normal[i][j])
            coord_product[i][j] = x * y * z
            norms_one_hot, norms_octiles = assign_octile(x, y, z, norms_one_hot, norms_octiles, i, j)
    return norms_one_hot


""" Main Explanation Generation Function """

def generate_explanations(task, multi_task_model, test_data, test_pred_full, image_np_depth_dec, norms_one_hot, k, device):
    # if task == "multi":
    #     cam_image = show_grad_cam_all_tasks(multi_task_model, test_data, test_pred_full)
    #     save_images(cam_image, "cam_all_tasks_3:" + str(k))
    if task == "SurNorm":
        counts = np.zeros(8)
        explanation_images = [] # in form ((image, task, octant),...)

        prev_clas_time = time.time()
        for norm_class in range(8):
            mask_one_float, mask = show_mask_depth_and_norms(test_data, norms_one_hot, norm_class)
            # Apply mask_one_float to test_data, blacking out the relevant pixels
            mask_one_float_tensor = torch.from_numpy(mask_one_float).unsqueeze(0).unsqueeze(0).to(device)
            resized_mask = F.interpolate(mask_one_float_tensor, size=(480, 640), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            # Apply the resized mask to test_data
            masked_test_data = test_data * resized_mask.unsqueeze(0)
            # Show the masked image
            masked_image = masked_test_data[0].cpu().squeeze().permute(1, 2, 0).numpy()
            masked_image = (masked_image - masked_image.min()) / (masked_image.max() - masked_image.min())
            plt.imshow(masked_image)
            plt.title(f'Masked Image for Norm Class {norm_class}')
            # plt.show()
            # save image
            plt.imsave(f'poster_images/masked_image{k}_{norm_class}.png', masked_image)

            # save_images(mask, "mask_image_surface_norms_pre_normalization" + str(norm_class) + ":" + str(k))
            cam_image, grayscale_cam = show_seg_grad_cam(multi_task_model, test_data, norm_class, mask_one_float, device, k,task_type="normals")
            # save_images(cam_image, "cam_image_surface_norms_pre_normalization" + str(norm_class) + ":" + str(k))
            create_cam_time = time.time()
            # print("Time to create CAM: ", round(create_cam_time - prev_clas_time, 3), " seconds", end=" ")

            if mask_one_float.sum() > 0 and grayscale_cam.sum() > 0:
                # road_images_percentile_and_class.extend(generate_ROAD_inputs(test_data, grayscale_cam, targets, multi_task_model, norm_class))
                
                class_cam_image = np.array(grayscale_cam)
                explantion_on_image = np.array(cam_image)

                np.save("./xplanations_new/Surface Normals/X_Image" + str(k) + "_norm" + str(norm_class), class_cam_image)
                np.save("./xplanations_new/Surface Normals/XonI_Image" + str(k) + "_norm" + str(norm_class), explantion_on_image)

                # Add grad-cam image to list
                explanation_images.append((grayscale_cam, task, norm_class))

            one_class_time = time.time()
            print("Time to create explanations for class", norm_class, ":", round(one_class_time - prev_clas_time, 1), " seconds")
            prev_clas_time = one_class_time

        return explanation_images



""" Main Function """

def voj_explanation_generator(test_loader, multi_task_model, device, task, num_images_to_gen_explanation = 200):
    # tasks = ["SemSeg", "Depth", "SurNorm", "multi"]
    # modes = ["Generation", "Evaluation"]
    tasks = [task] # only select the tasks that u wanna generate explantions for
    
    # mode = modes[0]
    # seg_classes = ["Bed", "Books", "Ceiling", "Chair", "Floor", "Furniture", "Objects", "Picture", "Sofa", "Table", "TV", "Wall", "Window"]
    surface_norm_classes = [(-1,-1,-1), (-1, 1, -1), (1, -1, -1), (1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)] # signs of x, y and z
    print("Creating Explanations...")
    begin = time.time()

    # evaluating test data
    multi_task_model.eval()
    road_images_results = [] # List of tuples (perc, cat_class, metric, input_number)
    torch.manual_seed(6277)


    with torch.no_grad():  # operations inside don't track history
        test_dataset = iter(test_loader)
        
        for k in range(num_images_to_gen_explanation): # Iterate through images
            batch_start = time.time()
            test_data, test_label, test_depth, test_normal = next(test_dataset)
            test_data, test_label = test_data.to(device), test_label.long().to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            # create image labels
            input_image = test_data.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            seg_label = test_label.cpu().numpy().squeeze(0)
            depth_label = test_depth.cpu().numpy().squeeze(0).squeeze(0)
            norm_label = test_normal.cpu().numpy().squeeze(0).transpose(1, 2, 0)
            # normalize the normal labels between 0 and 1
            norm_label = (norm_label + 1) / 2
            
            # save the original image and labels here
            plt.imsave("poster_images/Image_Input"+str(k)+".png", input_image)
            plt.imsave("poster_images/Seg_Label"+str(k)+".png", seg_label)

            plt.imsave("poster_images/Depth_Label"+str(k)+".png", depth_label)
            plt.imsave("poster_images/Norm_Label"+str(k)+".png", norm_label)
            

            test_pred_full, _ = multi_task_model(test_data)

            # show_outputs(test_data, test_pred_full)
            # show_image(test_pred_full[0], "SemSeg")

            for task in tasks: # Iterate through tasks
                start_task = time.time()
                print("Time till start of task: ", round(start_task - begin, 3), " seconds")
                """ Preprocess the regresion outputs """
                image_np_depth_quin, norms_one_hot = None, None
                if task == "SurNorm": # Get octants
                    norms_one_hot = preprocess_surface_normals(test_pred_full)
                    
                pre_pro_time = time.time()
                print("Time to pre-processing: ", round(pre_pro_time - start_task, 3), " seconds")

                """ Generate Explanations """
                explanations_generated = generate_explanations(task, multi_task_model, test_data, test_pred_full, image_np_depth_quin, norms_one_hot, k, device)
            
            batch_end = time.time()
            print("Time for one batch: ", round(batch_end - batch_start, 3), " seconds")