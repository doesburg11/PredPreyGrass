from PIL import Image
import os

# Specify the directory containing images
image_folder = '/home/doesburg/marl/PredPreyGrass/assets/images'

# Get all file names in the directory
image_files = os.listdir(image_folder)

# Sort the images by name
image_files.sort()

# Create a list to hold the images
images = []

# Read each file in the folder
for image_file in image_files:
    if image_file.endswith('.png'):  # or '.jpg', '.jpeg', etc.
        # Open the image file and append it to the list
        images.append(Image.open(os.path.join(image_folder, image_file)))

# Save the images as an animated gif
images[0].save('predprey.gif', save_all=True, append_images=images[1:], loop=0, duration=500)