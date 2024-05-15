"""
Created on Mon Mar 25 15:45:38 2024

@author: fifon
"""



import os
import cv2
import numpy as np
import albumentations as A

def create_augmentation_functions():
    
    augmentation_functions = []
    
    # Define a list of possible augmentations
    # Dataset - Car_Bike
    augmentations = [
        A.HorizontalFlip(p=0.5),                
        A.VerticalFlip(p=0.5),                  
        A.RandomRotate90(p=0.5),  
        A.Rotate(limit=30, p=0.5), 
        A.Flip(p=0.5),                           
        A.Transpose(p=0.5),                      
        A.GaussNoise(p=0.5),                     
        A.RandomBrightnessContrast(p=0.5),     
        A.RandomGamma(p=0.5),                   
        A.Blur(p=0.5),                          
        A.HueSaturationValue(p=0.5),            
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),  
        A.ToGray(p=0.5),                        
        A.CLAHE(p=0.5),                         
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),  
        A.GridDistortion(p=0.5)
    ]
    
    
    # Define a list of possible augmentations
    # Dataset - Histology_images
    # augmentations = [
    #     A.HorizontalFlip(p=0.5),  # Flip the image horizontally
    #     A.VerticalFlip(p=0.5),  # Flip the image vertically
    #     A.Rotate(limit=15, p=0.5),  # Limit rotation to +/- 15 degrees to preserve structure
    #     A.GaussNoise(p=0.25),  # Add Gaussian noise with a lower probability
    #     A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1),  # Slightly adjust brightness and contrast
    #     A.CLAHE(p=0.5),  # Enhances local contrast, potentially useful
    #     A.CoarseDropout(max_holes=4, max_height=4, max_width=4, p=0.25),  # Simulate occlusions with less intensity
    # ]
    
    # Randomly select 5 augmentations from the list
    for _ in range(5):
        augmentation = np.random.choice(augmentations)
        augmentation_functions.append(augmentation)
    
    return augmentation_functions
    
   
def augment_dataset_subset(input_subset, output_dir, augmentation_functions, augmentation_factor):
    # Create output directories for each class if they don't exist
    dataset_classes = input_subset.dataset.classes
    for class_name in dataset_classes:
        class_output_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

    for idx in range(len(input_subset)):
        # Get the sample from the subset
        img, label = input_subset[idx]  # Assuming the sample is a tuple with image data and label

        try:
            img = img.numpy().transpose(1, 2, 0)  # Convert from tensor to numpy array and transpose dimensions
            img = (img * 255).astype(np.uint8)  # Convert from float32 to uint8

            # Apply augmentation techniques multiple times
            for i in range(augmentation_factor):
                augmented_img = img.copy()

                # Randomly select a subset of augmentation functions
                num_augmentations = np.random.randint(2, 4)  
                selected_augmentations = np.random.choice(augmentation_functions, num_augmentations, replace=True)

                # Create a pipeline of augmentations
                augmentation_pipeline = A.Compose(selected_augmentations)

                # Apply the pipeline of augmentations to the image
                augmented = augmentation_pipeline(image=augmented_img)
                augmented_img = augmented["image"]

                # Save the augmented image in the corresponding class folder
                class_name = dataset_classes[label]
                class_output_dir = os.path.join(output_dir, class_name)
                output_path = os.path.join(class_output_dir, f"sample_{idx}_{i}.jpg")
                cv2.imwrite(output_path, augmented_img)

        except Exception as e:
            print(f"Error processing sample {idx}. Error message: {str(e)}")

    print("Augmentation completed.")


    

