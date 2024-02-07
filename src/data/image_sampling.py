import os
import shutil
import random

# Preparation
random.seed(0)
os.chdir('c:\\Users\\andre\\iCloudDrive\\Dokumente\\Master Data Science\\3. Semester (WS 23)\\DS500-Data_Science_Project\\BikeWayFinder\\data')

# Source and destination folders
source_folder = 'streetview_images/Stuttgart/edges'
destination_folder = 'streetview_images/Stuttgart/sampled_edges'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# List all image files in the source folder
image_files = [file for file in os.listdir(source_folder) if file.endswith(('.jpg', '.png', '.jpeg'))]

# Randomly sample 1000 images
num_samples = min(1000, len(image_files))
sampled_images = random.sample(image_files, num_samples)

#for image in sampled_images:
#    source_path = os.path.join(source_folder, image)
#    destination_path = os.path.join(destination_folder, image)
#    shutil.copy(source_path, destination_path)
    
    
#### Create .csv for Tivian

# Get all jpg files from the directory
file_names = [f for f in os.listdir(destination_folder) if f.endswith('.jpg')]

# Sort the file names based on the cluster number (extracted from the file name)
file_names.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Create the CSV lines
csv_lines = [f"{int(file_name.split('_')[-1].split('.')[0])};{index + 1};{file_name};;;;;4" for index, file_name in enumerate(file_names)]

# First few lines for verification
csv_lines[:5]

# Write the CSV file
csv_file_path = "streetview_images/Stuttgart/sampled_edges.csv"  # replace with your desired file path
with open(csv_file_path, 'w') as file:
    for line in csv_lines:
        file.write(line + '\n')
