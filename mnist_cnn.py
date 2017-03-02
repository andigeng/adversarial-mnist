import os
import tensorflow as tf
import cnn_functions as cf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data



CURR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CURR_DIR, "MNIST_data")
SAVE_DIR = os.path.join(CURR_DIR, "saved_models/model.ckpt")

# Import data
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Create the model
# The following architecture is outlined in the 'Deep MNIST for experts' tutorial.
x = tf.placeholder(tf.float32, [None, 784])

# First convolutional layer and pool
W_conv1 = cf.weight_variable([5, 5, 1, 32])
b_conv1 = cf.bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(cf.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = cf.max_pool_2x2(h_conv1)

# Second convolutional layer and pool
W_conv2 = cf.weight_variable([5, 5, 32, 64])
b_conv2 = cf.bias_variable([64])
h_conv2 = tf.nn.relu(cf.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = cf.max_pool_2x2(h_conv2)

# Fully connected layer
W_fc1 = cf.weight_variable([7 * 7 * 64, 1024])
b_fc1 = cf.bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = cf.weight_variable([1024, 10])
b_fc2 = cf.bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

prediction = tf.argmax(y_conv,1)
target = tf.argmax(y_, 1)

is_prediction_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_prediction_correct, tf.float32))

sess = tf.Session()
saver = tf.train.Saver()


def train_model(save=True, steps=20000, batch_size=50):
    """ 
    Train the tensorflow model. The training procedure is outlined in the 'Deep MNIST for experts' 
    tutorial.
    """
    print("Training model.")
    for step in range(steps):
        batch = mnist.train.next_batch(batch_size)
        if step%100 == 0:
            print("Step: {}".format(step))
            feed_dict = {x:batch[0], y_: batch[1], keep_prob: 1.0}
            sample_accuracy(feed_dict, to_print=True, test_type='training')
            
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    testing_accuracy(to_print=True)
    if save: saver.save(sess, SAVE_DIR)


def load_model(directory):
    """ Load a past session in. This sets all the weights equal to the previous session. """
    print("Loading model.")
    with sess.as_default():
        # Restore model from disk.
        saver.restore(sess, directory)
        print("Model loaded.")


def get_mistakes():
    """ Returns images, labels, and predicted values of the misclassified test elements.

    Returns:
        images: a N by 784 numpy array representing N images
        labels: a N by 10 numpy array representing the correct labels
        predictions: a N by 10 numpy array repersenting the incorrect predictions
    """
    images = []
    labels = []
    predictions = []

    for idx in range(1000):
        image, label = get_test_element(idx)
        pred = prediction.eval(feed_dict={x:image, keep_prob:1.0})
        if label[0][pred] != 1:
            # Append mistaken images
            images.append(image[0])
            # Append mistaken target labels
            labels.append(label[0])
            # Append predicted labels
            incorrect_prediction = np.zeros(10)
            incorrect_prediction[pred] = 1
            predictions.append(incorrect_prediction)

    images, labels, predictions = np.array(images), np.array(labels), np.array(predictions)
    return images, labels, predictions


def get_test_element(idx):
    image = mnist.test.images[idx].reshape((-1, 784))
    label = mnist.test.labels[idx].reshape((-1, 10))
    return image, label


def gen_adverarial():
    """  """
    pass


def plot_image(image, label, prediction):
    """
    Input: 
        image: length 784 numpy array, representing a 28x28 image
        label: integer
        prediction: integer
    """
    print('plotting')
    figure, axes = plt.subplots()
    image = image.reshape((28,28))
    axes.imshow(image, cmap='gray_r')
    axes.set_xlabel("Actual: {}, Predicted: {}".format(label, prediction))
    axes.set_xticks([])
    axes.set_yticks([])
    plt.show()


def testing_accuracy(to_print=True):
    """ Run through the test set and return the accuracy, with an option to print. """
    feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
    return sample_accuracy(feed_dict, to_print=to_print, test_type='test') 


def sample_accuracy(feed_dict, to_print=True, test_type='training'):
    """ Run through the given set and return the accuracy, with an option to print. """
    accuracy_ratio = accuracy.eval(feed_dict)
    if to_print: print("{} accuracy {}".format(test_type, accuracy_ratio))
    return accuracy_ratio



def main():
    with sess.as_default():
        tf.global_variables_initializer().run()
        load_model(SAVE_DIR)
        images, labels, predictions = get_mistakes()
        for k in range(3):
            to_plot = images[k], np.argmax(labels[k]), np.argmax(predictions[k])
            plot_image(*to_plot)
    sess.close()


if __name__ == '__main__':
    main()