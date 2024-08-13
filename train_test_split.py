import os
import random
import shutil

# Set random seed for reproducibility
random.seed(42)

# Define paths
main_folder = '/hpctmp/pbs_dm_stage/access_temp_stage/e1100476/Dataset/retina images/APTOS-2019 dataset/colored_images'
output_folder = '/hpctmp/pbs_dm_stage/access_temp_stage/e1100476/Dataset/retina images/linprobe'  # Folder where train and val folders will be created
train_folder = os.path.join(output_folder, 'train')
val_folder = os.path.join(output_folder, 'val')

# Create train and val directories if they don't exist
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

if not os.path.exists(val_folder):
    os.makedirs(val_folder)

# Get subdirectories (categories)
categories = [d for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]

for category in categories:
    # Paths for category folders in train and val directories
    train_category_folder = os.path.join(train_folder, category)
    val_category_folder = os.path.join(val_folder, category)
    
    if not os.path.exists(train_category_folder):
        os.makedirs(train_category_folder)
    
    if not os.path.exists(val_category_folder):
        os.makedirs(val_category_folder)
    
    # Get all image files in the category folder
    category_folder = os.path.join(main_folder, category)
    images = [f for f in os.listdir(category_folder) if os.path.isfile(os.path.join(category_folder, f))]
    
    # Shuffle and split the images
    random.shuffle(images)
    split_point = int(0.8 * len(images))
    
    train_images = images[:split_point]
    val_images = images[split_point:]
    
    # Move images to train folder
    for image in train_images:
        shutil.move(os.path.join(category_folder, image), os.path.join(train_category_folder, image))
    
    # Move images to val folder
    for image in val_images:
        shutil.move(os.path.join(category_folder, image), os.path.join(val_category_folder, image))

print("Dataset split into train and val folders.")
