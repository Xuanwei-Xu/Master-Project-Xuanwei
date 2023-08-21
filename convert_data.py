import os
import glob
from PIL import Image
# Define the paths for original dataset
data_path = "brain_dataset"

# Define the paths for converted dataset
convert_path = "brain_dataset_24b"
# Search for files that match a specified pattern.
image_path = glob.glob(data_path + "/*/*")

'''Create a list of new class paths by appending the class names to the converted dataset path
using a list comprehension and print the list of image paths retrieved from the original dataset.'''
new_class_path = [os.path.join(convert_path, im_class) for im_class in os.listdir(data_path)]
print(image_path)

# Create directories for each class in the converted dataset if they don't already exist.
for path in new_class_path:
    if not os.path.exists(path):
        os.makedirs(path)

# Open each image in the path of original dataset and convert them to RGB format.
# Finally, save them to converted dataset path instead of original dataset.
for image_path in image_path:
    image = Image.open(image_path)
    image = image.convert("RGB")
    image.save(image_path.replace("brain_dataset", "brain_dataset_24b"))
