from fastai.vision.all import *

#Download and decompress the dataset
# The dataset is a collection of images of cats and dogs
# The images are stored in a folder called 'images'
# The dataset is downloaded from the internet using the URLs.PETS function
path = untar_data(URLs.PETS) / 'images'
#We need a way to label our images as dogs or cats.

def is_cat(x):
    return x[0].isupper() # Check if the first letter is uppercase

# Create DataLoaders
dls = ImageDataLoaders.from_name_func(
    path,                   # Path to images
    get_image_files(path),  # List of image files
    valid_pct=0.2,          # 20% validation set
    seed=42,                # Random seed
    label_func=is_cat,      # Labeling function (True=cat, False=dog)
    item_tfms=Resize(192)   # Resize images to 192x192
)

# Verify it works
dls.show_batch()
#lets train our model using resnet18(small and fast)
learn = vision_learner(dls,resnet18,metrics=error_rate)

learn.fine_tune(3)

from pathlib import Path

# Ensure the file is saved in the desired directory
export_path = Path('./cats_dogs.pkl')  # Save in the current working directory
learn.export(export_path)
# The model is saved to a file called 'cats_dogs.pkl'
#Observations
#the head cleaning is done automatically
# The model is trained using the resnet18 architecture
# The model is trained using the Adam optimizer
# The model is trained using the default learning rate
# The model is trained using the default batch size
# The model is trained for 3 epochs(cycles through the training data 3 times)
