

# Fashion MNIST Classification using JAX and Flax

This project aims to classify images from the Fashion MNIST dataset using a Convolutional Neural Network (CNN) implemented with JAX and Flax.


## Overview

The Fashion MNIST dataset comprises grayscale images of 10 fashion categories. This project uses a CNN to classify these images into their respective categories.

## Dependencies

- JAX
- Flax
- idx2numpy
- matplotlib
- TensorFlow
- scikit-learn
- optax

## Dataset

The dataset used is the Fashion MNIST dataset, which contains 60,000 training images and 10,000 test images. Each image is 28x28 pixels in grayscale. The dataset is loaded and preprocessed for training.

## Model Architecture

The CNN model consists of:
- Two convolutional layers with ReLU activation.
- A dense layer for classification.

## Training and Evaluation

The model is trained using mini-batch gradient descent with the Adam optimizer. The cross-entropy loss is used as the loss function. After training, the model is evaluated on the test dataset using various classification metrics.

## Results

The trained model achieves competitive accuracy on the test dataset. Detailed results, including a confusion matrix and classification report, are provided in the code.

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Run the provided Python script to train and evaluate the model.


