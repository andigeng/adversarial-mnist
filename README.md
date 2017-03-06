# adversarial-mnist
These are scripts that let you train a simple convolutional neural network to classify [MNIST](http://yann.lecun.com/exdb/mnist/) digits, and then create [adversarial images](http://karpathy.github.io/2015/03/30/breaking-convnets/) to fool the model. The architecture of the network is outlined [here](https://www.tensorflow.org/get_started/mnist/pros).

## Getting Started
Clone or download the repository to a local folder. Make sure you have Tensorflow 1.0.0, Numpy, Matplotlib, and Python 3.5.2.
### Creating and saving the adversarial examples
This will create 10 adversarial images of the number two which are misclassified as six. See the end result [here](https://cloud.githubusercontent.com/assets/16085833/23596234/8e4bf90a-01f6-11e7-9db2-9e34b69cc8fc.png).
```
python adversarial_generator.py
```
### Making your own adversarial examples
Within adversarial_generator.py:
``` python
original_number = 2
target_number = 6
quantity = 10
example_arr = generate_adversarial_images(original_num, target_num, quantity)
plot_multiple_comparisons(*example_arr)
```
### Training the original model
There is a pre-trained model included, but you can retrain it if you wish.
```
python mnist_cnn.py
```