"""
Single script to train TuberculosisNet model
Combines dataset preparation, dependency check, and training logic
"""

import os
import sys
import argparse
import subprocess
import shutil
import random
import numpy as np
import csv

# TensorFlow import and compatibility
try:
    import tensorflow as tf
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        tf = tf.compat.v1
        tf.disable_eager_execution()
except ImportError:
    print("Error: TensorFlow not installed. Please install with: pip install tensorflow")
    sys.exit(1)

# Constants
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 1
BATCH_SIZE_TEST = 1
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 2
TRAIN_CSV_PATH = 'train_split.csv'
VAL_CSV_PATH = 'val_split.csv'
TEST_CSV_PATH = 'test_split.csv'
AUGMENTATION_CHANCE = 1.0
avg_train_intensity = (0.531459229, 0.531459229, 0.531459229)
avg_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
black_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3))
for r in range(len(avg_mask)):
    for c in range(len(avg_mask[0])):
        if r < 45 and (c < 45 or c > 178):
            avg_mask[r][c] = avg_train_intensity
            black_mask[r][c] = (False, False, False)
        else:
            black_mask[r][c] = (True, True, True)

# Data pipeline
class TBNetDSI:
    def __init__(self, data_path='data/'):
        self.data_path = data_path
    def parse_function(self, filename, label):
        file_contents = tf.read_file(filename)
        img_decoded = tf.image.decode_image(file_contents, channels=3)
        img = tf.image.crop_to_bounding_box(img_decoded, 11, 11, 168, 202)
        img = tf.image.resize_images(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)/255.
        img = tf.where(black_mask, img, tf.zeros_like(img))
        min_value = tf.reduce_min(img)
        max_value = tf.reduce_max(img)
        img = tf.divide(tf.subtract(img, min_value), max_value)
        img = tf.math.add(img, avg_mask)
        return {'image': img,
            'label/one_hot': tf.one_hot(label, NUM_CLASSES),
            'label/value': label,
            'placeholder': tf.convert_to_tensor(1, dtype=tf.float32) }
    def parse_function_train(self, filename, label):
        file_contents = tf.read_file(filename)
        img_decoded = tf.image.decode_image(file_contents, channels=3)
        img = tf.image.crop_to_bounding_box(img_decoded, 11, 11, 168, 202)
        img = tf.image.resize_images(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)/255.
        if random.random() < AUGMENTATION_CHANCE:
            which_aug = random.randint(0,3)
            if which_aug == 0:
                img = tf.random_crop(img, [202,202,3])
                img = tf.image.resize_images(img, [224,224])
            elif which_aug == 1:
                img = tf.image.random_flip_left_right(img)
            elif which_aug == 2:
                img = tf.image.random_brightness(img, 0.1)
            elif which_aug == 3:
                img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.where(black_mask, img, tf.zeros_like(img))
        min_value = tf.reduce_min(img)
        max_value = tf.reduce_max(img)
        img = tf.divide(tf.subtract(img, min_value), max_value)
        img = tf.math.add(img, avg_mask)
        return {'image': img, 
            'label/one_hot': tf.one_hot(label, NUM_CLASSES),
            'label/value': label,
            'placeholder': tf.convert_to_tensor(1, dtype=tf.float32) }
    def get_split(self, csv_path, phase="train", num_shards=1, shard_index=0):
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

def eval(sess, graph, val_or_test, dataset, image_tensor, label_tensor, pred_tensor, loss_tensor):
    from sklearn.metrics import confusion_matrix
    y_test = []
    predictions = []
    num_evaled = 0
    total_loss = 0.0
    iterator = dataset.make_initializable_iterator()
    datasets = {}
    datasets[val_or_test] = {
        'dataset': dataset,
        'iterator': iterator,
        'gn_op': iterator.get_next(),
    }
    sess.run(datasets[val_or_test]['iterator'].initializer)
    while True:
        try:
            data_dict = sess.run(datasets[val_or_test]['gn_op'])
            images = data_dict['image']
            labels = data_dict['label/one_hot'].argmax(axis=1)
            pred = sess.run(pred_tensor, feed_dict={image_tensor: images})
            predictions.append(pred)
            y_test.append(labels)
            num_evaled += len(pred)
            if val_or_test == "val":
                # Ensure accumulation uses a scalar; sess.run may return ndarray
                batch_loss = sess.run(loss_tensor, feed_dict={image_tensor: images, label_tensor: labels})
                # Handle both scalar and array loss values
                batch_loss_array = np.asarray(batch_loss)
                if batch_loss_array.size == 1:
                    total_loss += float(batch_loss_array.reshape(()))
                else:
                    total_loss += float(np.mean(batch_loss_array))
        except tf.errors.OutOfRangeError:
            print("\tEvaluated {} images.".format(num_evaled))
            break
    if val_or_test == "val":
        print("Minibatch loss=", "{:.9f}".format(float(total_loss)))
    matrix = confusion_matrix(np.array(y_test), np.array(predictions))
    matrix = matrix.astype('float')
    print(matrix)
    class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
    print('Sens Normal: {0:.3f}, Tuberculosis: {1:.3f}'.format(class_acc[0],class_acc[1]))
    ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
    print('PPV Normal: {0:.3f}, Tuberculosis {1:.3f}'.format(ppvs[0],ppvs[1]))

def main():
    parser = argparse.ArgumentParser(description='Single TuberculosisNet Training Script')
    parser.add_argument('--weightspath', default='TB-Net', type=str, help='Path to checkpoint folder')
    parser.add_argument('--metaname', default='model_train.meta', type=str, help='Name of ckpt meta file')
    parser.add_argument('--ckptname', default='model', type=str, help='Name of model ckpt')
    parser.add_argument('--datapath', default='data/', type=str, help='Root folder containing the dataset')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--savepath', default='models/', type=str, help='Folder for models to be saved in')
    args = parser.parse_args()
    LEARNING_RATE = args.lr
    OUTPUT_PATH = args.savepath
    EPOCHS = args.epochs
    VALIDATE_EVERY = 5
    INPUT_TENSOR = "image:0"
    LABEL_TENSOR = "classification/label:0"
    LOSS_TENSOR = "add:0"
    PREDICTION_TENSOR = "ArgMax:0"
    dsi = TBNetDSI(data_path=args.datapath)
    train_dataset, train_dataset_size, train_batch_size = dsi.get_train_dataset()
    val_dataset, _, _ = dsi.get_validation_dataset()
    test_dataset, _, _ = dsi.get_test_dataset()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(args.weightspath, args.metaname))
    graph = tf.get_default_graph()
    # Try to find tensors (support both original checkpoint and synthetic one)
    try:
        image_tensor = graph.get_tensor_by_name("image:0")
    except KeyError:
        # Check available operations if tensor not found
        print("Error: Input tensor 'image:0' not found.")
        print("Available operations sample:", [op.name for op in graph.get_operations()][:10])
        raise

    try:
        label_tensor = graph.get_tensor_by_name("classification/label:0")
    except KeyError:
        label_tensor = graph.get_tensor_by_name("label:0")

    try:
        pred_tensor = graph.get_tensor_by_name("ArgMax:0")
    except KeyError:
        try:
            pred_tensor = graph.get_tensor_by_name("classification/ArgMax:0")
        except KeyError:
            print("Error: Prediction tensor not found (tried 'ArgMax:0' and 'classification/ArgMax:0')")
            raise

    try:
        loss_tensor = graph.get_tensor_by_name("add:0")
    except KeyError:
        try:
            loss_tensor = graph.get_tensor_by_name("classification/add:0")
        except KeyError:
            print("Error: Loss tensor not found (tried 'add:0' and 'classification/add:0')")
            raise
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss_tensor)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver.restore(sess, os.path.join(args.weightspath, args.ckptname))
    save_path = os.path.join(OUTPUT_PATH, "Baseline/TB-Net")
    os.makedirs(save_path, exist_ok=True)
    saver.save(sess, save_path)
    print('Saved baseline checkpoint to {}.'.format(save_path))
    print('Baseline eval:')
    eval(sess, graph, "test", test_dataset, image_tensor, label_tensor, pred_tensor, loss_tensor)
    print('Training started')
    iterator = train_dataset.make_initializable_iterator()
    datasets = {}
    datasets['train'] = {
        'dataset': train_dataset,
        'iterator': iterator,
        'gn_op': iterator.get_next(),
    }
    sess.run(datasets['train']['iterator'].initializer)
    num_batches = train_dataset_size // train_batch_size
    progbar = tf.keras.utils.Progbar(num_batches)
    for epoch in range(EPOCHS):
        for i in range(num_batches):
            data_dict = sess.run(datasets['train']['gn_op'])
            batch_x = data_dict['image']
            batch_y = data_dict['label/one_hot'].argmax(axis=1)
            sess.run(train_op, feed_dict={image_tensor: batch_x, label_tensor: batch_y})
            progbar.update(i+1)
        if epoch % VALIDATE_EVERY == 0:
            eval(sess, graph, "val", val_dataset, image_tensor, label_tensor, pred_tensor, loss_tensor)
            saver.save(sess, os.path.join(OUTPUT_PATH, "Epoch_" + str(epoch), "TB-Net"), global_step=epoch+1, write_meta_graph=False)
            print('Saving checkpoint at epoch {}'.format(epoch + 1))
    print("Optimization Finished!")

if __name__ == '__main__':
    main()
