import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image
import time
import copy

from NST_main import (
    image_loader,
    imshow,
    ContentLoss,
    gram_matrix,
    StyleLoss,
    Normalization,
    get_style_model_and_losses,
    get_input_optimizer,
    run_style_transfer
)

#%% 5_A

# Define the path to your images folder
images_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_5/"

# Create a results folder if it doesn't exist
results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_5/results1/"


if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Get a list of all image files in the folder
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpeg')])

      
# Iterate through pairs of images and apply style transfer
for i in range(len(image_files)):
    for j in range(i + 1, len(image_files)):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")
   
    


#%% 5_B

results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_5/results2/"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
    
# Iterate through pairs of images and apply style transfer
for j in range(len(image_files)):
    for i in range(j + 1, len(image_files)):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")
   
    
#%% 10_A

# Define the path to your images folder
images_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_10/"

# Create a results folder if it doesn't exist
results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_10/results1/"


if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Get a list of all image files in the folder
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpeg')])

      
# Iterate through pairs of images and apply style transfer
for i in range(len(image_files)):
    for j in range(i + 1, len(image_files)):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")
   
    
#%% 10_B

results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_10/results2/"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
    
# Iterate through pairs of images and apply style transfer
for j in range(len(image_files)):
    for i in range(j + 1, len(image_files)):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")
   
#%% 20_A


# Define the path to your images folder
images_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_20/"

# Create a results folder if it doesn't exist
results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_20/results1/"


if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Get a list of all image files in the folder
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpeg')])

      
# Iterate through pairs of images and apply style transfer
for i in range(len(image_files)):
    for j in range(i + 1, len(image_files)):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")

#%% CAR_20_B

results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_20/results2/"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
    
# Iterate through pairs of images and apply style transfer
for j in range(len(image_files)):
    for i in range(j + 1, len(image_files)):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")
        
#%% CAR_40_A


# Define the path to your images folder
images_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_40/"

# Create a results folder if it doesn't exist
results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_40/results1/"


if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Get a list of all image files in the folder
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpeg')])


max_number_40 = 24 #24*23 = 552 - vymazat 32

# Iterate through pairs of images and apply style transfer
for i in range(max_number_40):
    for j in range(i + 1, max_number_40):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")

#%% CAR_40_B

results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_40/results2/"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
    
# Iterate through pairs of images and apply style transfer
for j in range(max_number_40):
    for i in range(j + 1, max_number_40):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")
        

#%% CAR_80_A

# Define the path to your images folder
images_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_80/"

# Create a results folder if it doesn't exist
results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_80/results1/"


if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Get a list of all image files in the folder
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpeg')])


max_number_80 = 23 #23*22 = 506 - vymazať 26

# Iterate through pairs of images and apply style transfer
for i in range(max_number_80):
    for j in range(i + 1, max_number_80):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")
        
#%% CAR_80_B

results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_80/results2/"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
    
# Iterate through pairs of images and apply style transfer
for j in range(max_number_80):
    for i in range(j + 1, max_number_80):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")
        
#%% CAR_160_A


# Define the path to your images folder
images_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_160/"

# Create a results folder if it doesn't exist
results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_160/results1/"


if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Get a list of all image files in the folder
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpeg')])


max_number_160 = 21 #21*20 = 420 - vymazať 20

# Iterate through pairs of images and apply style transfer
for i in range(max_number_160):
    for j in range(i + 1, max_number_160):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")

#%% CAR_160_B

results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_160/results2/"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
    
# Iterate through pairs of images and apply style transfer
for j in range(max_number_160):
    for i in range(j + 1, max_number_160):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")
        
#%% CAR_320_A


# Define the path to your images folder
images_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_320/"

# Create a results folder if it doesn't exist
results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_320/results1/"


if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Get a list of all image files in the folder
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpeg')])


max_number_320 = 16 #16*15 = 240 

# Iterate through pairs of images and apply style transfer
for i in range(max_number_320):
    for j in range(i + 1, max_number_320):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")

#%% CAR_320_B

results_folder = "./dataset/my_lung_cancer_1000/class_train_experiment_NST/lung_scc_320/results2/"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
    
# Iterate through pairs of images and apply style transfer
for j in range(max_number_320):
    for i in range(j + 1, max_number_320):
    
        start_time = time.time()
        
        # Load style and content images
        style_img = image_loader(os.path.join(images_folder, image_files[i]))
        content_img = image_loader(os.path.join(images_folder, image_files[j]))
        
        assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"
        
        input_img = content_img.clone()
        
       
        cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Run style transfer
        output, style_loss_values, content_loss_values = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                          content_img, style_img, input_img)
       
        # Flatten the tensors
        flat_tensor1 = torch.flatten(content_img)
        flat_tensor2 = torch.flatten(style_img)
        flat_tensor3 = torch.flatten(output)
    
        # Compute Euclidean distance content-output
        euclidean_distance_content_output = torch.dist(flat_tensor1, flat_tensor3, p=2)
        euclidean_distance_content_output = 1 / (1 + euclidean_distance_content_output.item())
    
        # Compute Euclidean distance style-output
        euclidean_distance_style_output = torch.dist(flat_tensor2, flat_tensor3, p=2)
        euclidean_distance_style_output = 1 / (1 + euclidean_distance_style_output.item())
    
        print("Euclidean Similarity Content - Output:", euclidean_distance_content_output)
        print("Euclidean Similarity Style - Output:", euclidean_distance_style_output)
        
    
        # Save the resulting image
        result_filename = f"result_{i+1}_and_{j+1}.jpeg"
        output_path = os.path.join(results_folder, result_filename)
        save_image(output, output_path)
    
        print(f"Saved result {result_filename}")
        
        end_time = time.time() 
        elapsed_time_seconds = end_time - start_time
        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
    
        print(f"Time taken for iteration {i+1}: {int(elapsed_minutes)} minutes and {elapsed_seconds:.2f} seconds")