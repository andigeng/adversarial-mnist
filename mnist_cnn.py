import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


CURR_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CURR_DIR, 'MNIST_data')
SAVE_DIR = os.path.join(CURR_DIR, 'saved_models/model.ckpt')

# Import data
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.Session()
saver = tf.train.Saver()


def train_model(_):
    with sess.as_default():
        tf.global_variables_initializer().run()

        # Train model from scratch
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        test_accuracy()
        save_path = saver.save(sess, SAVE_DIR)

    sess.close()


def load_model(_):
    with sess.as_default():
        # Restore variables from disk.
        saver.restore(sess, SAVE_DIR)
        print("Model restored.")

        test_accuracy()

    sess.close()


def test_accuracy():
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def main():
    tf.app.run(train_model)
    # tf.app.run(load_model)



if __name__ == '__main__':
    main()