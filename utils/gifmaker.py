#import imageio
import imageio.v2 as imageio
import os

# Specify the directory containing the images
image_directory = './assets/images'

# Get all file names in the directory
file_names = os.listdir(image_directory)

# Filter the file names for .png files
image_files = [file for file in file_names if file.endswith('.png')]

# Sort the image files (optional, use if images should be in a specific order)
image_files.sort()

# Create a list to hold the images
images = []

# Read each image file and add it to the images list
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    images.append(imageio.imread(image_path))

# Save the images as a GIF
imageio.mimsave('./assets/gif/predpreygrass1.gif', images)