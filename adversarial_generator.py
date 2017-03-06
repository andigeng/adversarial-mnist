"""Creates 10 adversarial images of twos that are misclassified as sixes, and plots them.

The architecture of the network is outlined in the 'Deep MNIST for experts' tutorial in the official 
Tensorflow docs. The adversarial images are created by first calculating the gradient of the 
input image with respect to the loss function. Then, we subtract this gradient value from our 
noise image. This is then combined with the original image, and then fed back into the network.
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import cnn_functions as cf
import utils


# Import data
mnist = utils.get_data()

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
        example_arr = generate_adversarial_images(original_num=2, 
                                                  target_num=6,
                                                  quantity=10)
        plot_multiple_comparisons(*example_arr)
    sess.close()


def load_model(directory):
    """ Load a past session in. This sets all the weights equal to the previous session. """
    print("Loading model.")
    # Restore model from disk.
    saver.restore(sess, directory)
    print("Model loaded.")


def generate_adversarial_images(original_num, target_num, quantity, max_steps=10000):
    """ Creates and returns three lists of images: originals, noise, and the adversarial images. """
    idx_arr = find_image_indices(original_num, quantity)
    image_arr, noise_arr, adversarial_arr = [], [], []
    
    for idx in idx_arr:
        image, _ = get_test_element(idx)
        _, noise, adversarial = generate_adversarial_image(image, target_num, max_steps)
        image_arr.append(image)
        noise_arr.append(noise)
        adversarial_arr.append(adversarial)

    saved_arr = np.array(image_arr + noise_arr + adversarial_arr)
    np.save('adversarial_examples/adversarial_grid', saved_arr)
    return image_arr, noise_arr, adversarial_arr


def generate_adversarial_image(image, target, max_steps, change_rate=1.0/4000):
    """ Creates and returns three numpy arrays: image, noise, and the adversarial iamge. """
    label = get_target_label(target)
    noise = np.zeros((1, 784))
    # Generates gradients with respect to loss function and the input image.
    gradient = tf.gradients(cross_entropy, [x])[0]

    for step in range(max_steps):
        adversarial_image = image + noise
        # We clip the values so that they are between 0 and 1
        np.clip(adversarial_image, 0.0, 1.0, out=adversarial_image)
        noise = adversarial_image - image

        # Evaluate the gradient of the adversarial input with respect to our loss
        feed_dict = {x: adversarial_image, y_: label, keep_prob: 1.0}
        gradient_value = sess.run(gradient, feed_dict)

        if is_adversarial(adversarial_image, target):
            # We stop the process when the adversarial image is correctly misclassified.
            # This prevents the adversarial image from looking too different than the original.
            print("Adversarial image found. {} steps taken.".format(step))
            break

        noise -= gradient_value * change_rate

    if not is_adversarial(adversarial_image, target):
        print("No adversarial image found. Try increasing max_steps or change_rate.")

    return image, noise, adversarial_image


def find_image_indices(target, quantity):
    """ Given a target label, and quantity, tries to find elements that match the criteria.
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


def plot_multiple_comparisons(image_arr, noise_arr, adversarial_arr):
    """ Given three lists of images, plots and saves a n x 3 image comparing them. """
    all_image_arr = image_arr + noise_arr + adversarial_arr
    num_images = len(image_arr)

    figure, axes = plt.subplots(num_images, 3)
    figure.set_size_inches(3, num_images, forward=True)
    
    for row, ax in enumerate(axes):
        for col, sub_ax in enumerate(ax):
            image_to_plot = all_image_arr[col*num_images + row].reshape((28,28))
            sub_ax.imshow(image_to_plot, cmap='gray_r', vmin=0.0, vmax=1.0, interpolation='nearest')
            sub_ax.set_aspect("auto")
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])

    figure.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig('adversarial_examples/adversarial_grid', dpi=112)
    plt.show()


def get_test_element(idx):
    """ Return the image and label at a given index within the test set. """
    image = mnist.test.images[idx].reshape((-1, 784))
    label = mnist.test.labels[idx].reshape((-1, 10))
    return image, label


def get_target_label(num):
    """ Returns the specified hot-encoded label, a numpy array. """
    label = np.zeros([1,10])
    label[0][num] = 1.0
    return label


def is_adversarial(image, target):
    return prediction.eval(feed_dict={x:image, keep_prob:1.0}) == target



if __name__ == '__main__':
    main()