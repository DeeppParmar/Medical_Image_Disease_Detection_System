import os
import numpy as np
import random
import csv

# Handle TensorFlow imports - support both TF 1.x and 2.x
try:
    import tensorflow as tf
    # Use TensorFlow 2.x compatibility mode if available
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        # TensorFlow 2.x - use compat.v1 for TF 1.x APIs
        tf = tf.compat.v1
    # TensorFlow 1.x - use as is
except ImportError:
    raise ImportError("TensorFlow not installed. Please install with: pip install tensorflow")

BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 1 # Must be 1
BATCH_SIZE_TEST = 1 # Must be 1
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 2

# Point these paths to the corresponding csv files
TRAIN_CSV_PATH = 'train_split.csv'
VAL_CSV_PATH = 'val_split.csv'
TEST_CSV_PATH = 'test_split.csv'

# Chance that a random augmentation will be performed on an image
AUGMENTATION_CHANCE = 1.0

# Create the average mask
avg_train_intensity = (0.531459229, 0.531459229, 0.531459229)

avg_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
black_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
for r in range(len(avg_mask)):
    for c in range(len(avg_mask[0])):
        # Hardcoded values for the corners.
        if r < 45 and (c < 45 or c > 178):
            avg_mask[r][c] = avg_train_intensity
            black_mask[r][c] = (False, False, False)
        else:
            black_mask[r][c] = (True, True, True)


class TBNetDSI:
    def __init__(self, data_path='data/'):
        self.data_path = data_path

    def parse_function(self, filename, label):
        # Use TF 1.x compatible file reading
        file_contents = tf.read_file(filename)
        img_decoded = tf.image.decode_image(file_contents, channels=3)
        # Crop 5% from each side, but 25% from the bottom
        img = tf.image.crop_to_bounding_box(img_decoded, 11, 11, 168, 202)
        img = tf.image.resize_images(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)/255.

        # Apply the mask onto the image, thus masking the corners
        img = tf.where(black_mask, img, tf.zeros_like(img))

        # Renormalize
        min_value = tf.reduce_min(img)  # min is 0 because of black masking
        max_value = tf.reduce_max(img)
        img = tf.divide(tf.subtract(img, min_value), max_value)

        # Next, apply the average mask to make the corners grayish instead of black
        img = tf.math.add(img, avg_mask)

        # Convert the label into one hot, return
        return {'image': img,
            'label/one_hot': tf.one_hot(label, NUM_CLASSES),
            'label/value': label,
            'placeholder': tf.convert_to_tensor(1, dtype=tf.float32) }

    # Includes random augmentation
    def parse_function_train(self, filename, label):
        # Use TF 1.x compatible file reading
        file_contents = tf.read_file(filename)
        img_decoded = tf.image.decode_image(file_contents, channels=3)
        # Crop 5% from each side, but 25% from the bottom
        img = tf.image.crop_to_bounding_box(img_decoded, 11, 11, 168, 202)
        img = tf.image.resize_images(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)/255.

        # Chance for augmentation
        if random.random() < AUGMENTATION_CHANCE:
            # Select an augmentation randomly to perform. randint is inclusive.
            which_aug = random.randint(0,3)
            if which_aug == 0:
                img = tf.random_crop(img, [202,202,3])
                img = tf.image.resize_images(img, [224,224])
            elif which_aug == 1:
                img = tf.image.random_flip_left_right(img)
            elif which_aug == 2:
                img = tf.image.random_brightness(img, 0.1)
            elif which_aug == 3:
                img = tf.image.random_contrast(img, 0.8, 1.2)  # Fixed: contrast needs valid range


        # Apply the mask onto the image, thus masking the corners
        img = tf.where(black_mask, img, tf.zeros_like(img))

        # Renormalize
        min_value = tf.reduce_min(img)
        max_value = tf.reduce_max(img)
        img = tf.divide(tf.subtract(img, min_value), max_value)

        # Next, apply the average mask to make the corners grayish instead of black
        img = tf.math.add(img, avg_mask)

        # Convert the label into one hot, return
        return {'image': img, 
            'label/one_hot': tf.one_hot(label, NUM_CLASSES),
            'label/value': label,
            'placeholder': tf.convert_to_tensor(1, dtype=tf.float32) }


    def get_split(self, csv_path, phase="train", num_shards=1, shard_index=0):
        '''
        Reads in the data corresponding to the split
        '''
        data_x = []
        data_y = []
        with open(os.path.join(csv_path), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                filepath, label = row[0], row[1]
                data_x.append(os.path.join(self.data_path, filepath))
                data_y.append(int(label))

        dataset = tf.data.Dataset.from_tensor_slices((np.array(data_x), np.array(data_y)))
        num_data = len(data_y)

        if (phase == "train"):
            dataset = dataset.repeat()
            dataset = dataset.shuffle(5000)
            dataset = dataset.map(map_func=self.parse_function_train)
            batch_size = BATCH_SIZE_TRAIN
        elif (phase == "val"):
            dataset = dataset.map(map_func=self.parse_function)
            batch_size = BATCH_SIZE_VAL
        else:
            dataset = dataset.map(map_func=self.parse_function)
            batch_size = BATCH_SIZE_TEST

        dataset = dataset.batch(batch_size=batch_size)
        if num_shards > 1:
            dataset = dataset.shard(num_shards, shard_index)
        return dataset, num_data // num_shards, batch_size

    def get_train_dataset(self, num_shards=1, shard_index=0):
        dataset, num_data, batch_size = self.get_split(TRAIN_CSV_PATH, "train", num_shards, shard_index)
        return dataset, num_data, batch_size

    def get_validation_dataset(self, num_shards=1, shard_index=0):
        dataset, num_data, batch_size = self.get_split(VAL_CSV_PATH, "val", num_shards, shard_index)
        return dataset, num_data, batch_size

    def get_test_dataset(self):
        dataset, num_data, batch_size = self.get_split(TEST_CSV_PATH, "test")
        return dataset, num_data, batch_size
