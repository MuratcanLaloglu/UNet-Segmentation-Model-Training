from sklearn.model_selection import train_test_split
import os
import pickle

# list of image paths and corresponding mask paths
image_paths = ['./data/truth/' + f for f in os.listdir('./data/truth/')]
mask_paths = ['./data/mask/' + f for f in os.listdir('./data/mask/')]

# Split the dataset into training and validation sets
train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)


with open('train_val_split.pkl', 'wb') as f:
    pickle.dump({
        'train_img_paths': train_img_paths,
        'val_img_paths': val_img_paths,
        'train_mask_paths': train_mask_paths,
        'val_mask_paths': val_mask_paths
    }, f)
