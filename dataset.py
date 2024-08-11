from PIL import Image
import os
import uuid

# Define the path to your images
parent_folder = '/hpctmp/pbs_dm_stage/access_temp_stage/e1100476/Dataset/retina images/Diabetic Retinopathy 2015 Data/colored_images/colored_images'

# Define the path to save resized images
output_folder = '/hpctmp/pbs_dm_stage/access_temp_stage/e1100476/Dataset/retina images/pretrain/Diabetic Retinopathy 2015 Data'

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all subdirectories and files
for subdir, _, files in os.walk(parent_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(subdir, file)
            unique_name = "{}.png".format(uuid.uuid4().hex) 
            save_path = os.path.join(output_folder, unique_name)
            with Image.open(file_path) as img:
                img.save(save_path)

print("Moving complete.")
