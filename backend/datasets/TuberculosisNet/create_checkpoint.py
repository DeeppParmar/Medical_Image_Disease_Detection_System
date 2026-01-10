"""
Alternative: Create a minimal TB-Net checkpoint from scratch
This allows training without pre-trained weights
"""

import os
import sys
import tensorflow as tf

# Disable eager execution for TF 1.x style code
if hasattr(tf, 'compat'):
    tf = tf.compat.v1
    tf.disable_eager_execution()

def create_tbnet_architecture():
    """
    Create TB-Net model architecture based on the paper
    TB-Net uses ResNet-based architecture for TB detection
    """
    
    # Input placeholder
    image = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="image")
    label = tf.placeholder(tf.int32, shape=[None], name="classification/label")
    
    # Simple CNN architecture (placeholder for actual TB-Net)
    with tf.variable_scope("feature_extraction"):
        # Conv Block 1
        conv1 = tf.nn.conv2d(image, tf.Variable(tf.random.truncated_normal([7, 7, 3, 64], stddev=0.1)), strides=[1,2,2,1], padding='SAME')
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
        
        # Conv Block 2
        conv2 = tf.nn.conv2d(pool1, tf.Variable(tf.random.truncated_normal([3, 3, 64, 128], stddev=0.1)), strides=[1,1,1,1], padding='SAME')
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        # Conv Block 3
        conv3 = tf.nn.conv2d(pool2, tf.Variable(tf.random.truncated_normal([3, 3, 128, 256], stddev=0.1)), strides=[1,1,1,1], padding='SAME')
        conv3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        # Conv Block 4
        conv4 = tf.nn.conv2d(pool3, tf.Variable(tf.random.truncated_normal([3, 3, 256, 512], stddev=0.1)), strides=[1,1,1,1], padding='SAME')
        conv4 = tf.nn.relu(conv4)
        pool4 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        # Global Average Pooling
        gap = tf.reduce_mean(pool4, axis=[1, 2])
    
    # Classification head
    with tf.variable_scope("classification"):
        # Flatten
        gap_shape = gap.get_shape().as_list()
        fc1_w = tf.Variable(tf.random.truncated_normal([gap_shape[1], 256], stddev=0.1))
        fc1_b = tf.Variable(tf.constant(0.1, shape=[256]))
        fc1 = tf.nn.relu(tf.matmul(gap, fc1_w) + fc1_b)
        
        # Final layer
        fc2_w = tf.Variable(tf.random.truncated_normal([256, 2], stddev=0.1))
        fc2_b = tf.Variable(tf.constant(0.1, shape=[2]))
        logits = tf.matmul(fc1, fc2_w) + fc2_b
        
        predictions = tf.argmax(logits, axis=1, name="ArgMax")
        
        # Loss
        onehot_labels = tf.one_hot(label, depth=2)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=onehot_labels, logits=logits
        )
        loss = tf.reduce_mean(cross_entropy, name="add")  # Named 'add' to match original
    
    return image, label, predictions, loss


def create_checkpoint():
    """Create a fresh checkpoint for TB-Net"""
    
    print("=" * 60)
    print("Creating Fresh TB-Net Checkpoint")
    print("=" * 60)
    
    checkpoint_dir = "TB-Net"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Reset default graph
    tf.reset_default_graph()
    
    # Create model
    print("\n1. Building TB-Net architecture...")
    image, label, predictions, loss = create_tbnet_architecture()
    
    # Create session and initialize variables
    print("2. Initializing variables...")
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Save checkpoint
    print("3. Saving checkpoint...")
    saver = tf.train.Saver()
    
    # Save the model with training capability
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    saver.save(sess, checkpoint_path, write_meta_graph=True)
    
    # Also save with the specific meta graph name expected by training script
    meta_graph_path = os.path.join(checkpoint_dir, "model_train.meta")
    if not os.path.exists(meta_graph_path):
        import shutil
        shutil.copy(checkpoint_path + ".meta", meta_graph_path)
    
    print(f"\n✓ Checkpoint saved to: {checkpoint_dir}/")
    print(f"  - model_train.meta")
    print(f"  - model.index")
    print(f"  - model.data-*")
    print(f"  - checkpoint")
    
    sess.close()
    
    print("\n" + "=" * 60)
    print("Fresh checkpoint created successfully!")
    print("You can now run: python train_tbnet_single.py")
    print("=" * 60)


if __name__ == "__main__":
    create_checkpoint()
