import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import cnn_functions as cf
import utils


# Import data
mnist = utils.get_data()

# Create the model
# The following architecture is outlined in the 'Deep MNIST for experts' tutorial.
x = tf.placeholder(tf.float32, [1, 784])

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
y_ = tf.placeholder(tf.float32, [1, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

prediction = tf.argmax(y_conv,1)
target = tf.argmax(y_, 1)
is_prediction_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_prediction_correct, tf.float32))

sess = tf.Session()
saver = tf.train.Saver()


def main():
    with sess.as_default():
        tf.global_variables_initializer().run()
        load_model(utils.SAVE_DIR)
        idx_arr = find_image_indices(2, 3)
        for idx in idx_arr:
            image, _ = get_test_element(idx)
            args = generate_adversarial_image(image, 6, max_steps=10000)
            plot_comparison(*args)
    sess.close()


def load_model(directory):
    """ Load a past session in. This sets all the weights equal to the previous session. """
    print("Loading model.")
    # Restore model from disk.
    saver.restore(sess, directory)
    print("Model loaded.")


def generate_adversarial_image(image, target, max_steps=10000, change_rate=1.0/4000, max_noise=0.25):
    """ Given an image and target, tries to create an image that will be misclassified to target.
    Input: 
        image: a numpy array of length 784 representing a 28x28 image
        target: integer
    Output:
        3 numpy arrays of length 784 representing the original image, noise, and adversarial image
    """
    label = get_target_label(target)
    noise = np.zeros((1, 784))
    # Generates gradients with respect to loss function and the input image.
    gradient = tf.gradients(cross_entropy, [x])[0]
    # This will hold the value of a gradient after being evaluated
    gradient_value = np.zeros((1, 784))

    for step in range(max_steps):
        # The adversarial image is the original image plus the noise image
        adversarial_image = image + noise
        # We clip the values so that they are between 0 and 1
        np.clip(adversarial_image, 0.0, 1.0, out=adversarial_image)
        # Evaluate the gradient of the adversarial input with respect to our loss
        gradient_value = sess.run(gradient, feed_dict={x: adversarial_image, y_: label, keep_prob:1.0})

        if step%100 == 0: 
            print(step)
        if prediction.eval(feed_dict={x:adversarial_image, keep_prob:1.0}) == target: 
            print(step)
            break
        # We subtract the gradient from the noise. This ensures that on the next pass, the 
        # chance of the model being fooled will increase.
        noise -= gradient_value*change_rate
        np.clip(noise, 0.0, max_noise, out=noise)

    new_prediction = prediction.eval(feed_dict={x:adversarial_image, keep_prob:1.0})[0]
    
    if new_prediction != target:
        print("No adversarial image found. Try increasing max_steps or change_rate.")

    return image, adversarial_image-image, adversarial_image


def find_image_indices(target, quantity):
    """ Given a target label, and quantity, tries to find elements that match the criteria.
    Input:
        target: integer from 0-9 representing what types of elements you want
        quantity: integer for how many you want
    Output:
        target_indices: list of indices representing the positions of the elements found
    """
    target_indices = []
    for idx in range(10000):
        label = mnist.test.labels[idx]
        if np.argmax(label) == target:
            target_indices.append(idx)
            if len(target_indices) == quantity: break
    return target_indices


def get_test_element(idx):
    image = mnist.test.images[idx].reshape((-1, 784))
    label = mnist.test.labels[idx].reshape((-1, 10))
    return image, label


def get_target_label(num):
    label = np.zeros([1,10])
    label[0][num] = 1.0
    return label


def plot_image(image, label, prediction):
    """ Plots an image with corresponding label and prediction.
    Input: 
        image: length 784 numpy array, representing a 28x28 imagez
        label: integer
        prediction: integer
    """
    print('plotting')
    figure, axes = plt.subplots()
    image = image.reshape((28,28))
    axes.imshow(image, cmap='gray_r', vmin=0.0, vmax=1.0)
    axes.set_xlabel("Actual: {}, Predicted: {}".format(label, prediction))
    axes.set_xticks([])
    axes.set_yticks([])
    plt.show()


def plot_comparison(image, noise, adversarial_image):
    """ Plots the original image, noise, and adversarial image side by side.
    Input: 
        3 numpy arrays of size 784, each representing a 28x28 image 
    """
    plots = [(image, 'original'), (noise, 'noise'), (adversarial_image, 'adversarial')]
    figure, axes = plt.subplots(1,3)
    for idx, ax in enumerate(axes):
        image_to_plot = plots[idx][0].reshape((28,28))
        ax.imshow(image_to_plot, cmap='gray_r', vmin=0.0, vmax=1.0)
        ax.set_xlabel("{} image".format(plots[idx][1]))
        ax.set_xticks([])
        ax.set_yticks([])
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


if __name__ == '__main__':
    main()