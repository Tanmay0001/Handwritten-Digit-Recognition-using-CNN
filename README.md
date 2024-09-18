# MNIST Handwritten Digit Recognition with CNN in Python using Keras

## Project Overview

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset using Python and Keras. The project includes a graphical user interface (GUI) that allows users to draw digits and get them recognized in real-time. The CNN model is compared with Multi-Layer Perceptrons (MLPs) to demonstrate the effectiveness of CNNs in image recognition tasks.

## Objectives

- **Understand CNNs**: Learn how CNNs can significantly improve digit recognition accuracy compared to Multi-Layer Perceptrons (MLPs).
- **Practical Implementation**: Gain hands-on experience with implementing and using CNNs in Python using the Keras library.

## Key Features

- **Real-Time Recognition**: Draw digits on a canvas and get instant predictions from the trained CNN model.
- **Clear Canvas**: Option to clear the canvas and start a new drawing.
- **High Accuracy**: Achieves high recognition accuracy with the trained CNN model.

## Key Design Considerations

- **Problem Type**: Multi-Class Classification (10 classes, representing digits 0 to 9)
- **Language**: Python
- **Deep Learning Package**: Keras
- **Dataset**: MNIST dataset (provided by Keras)
- **Model Architecture**: Convolutional Neural Network (CNN)

## CNN Model Architecture

The CNN model used in this project has the following architecture:

1. **Input Layer**: 28x28x1 matrix (grayscale image)
2. **Convolutional Layer**: 32 filters of size 3x3 with ReLU activation
3. **Max-Pooling Layer**: 2x2
4. **Convolutional Layer**: 64 filters of size 3x3 with ReLU activation
5. **Max-Pooling Layer**: 2x2
6. **Convolutional Layer**: 64 filters of size 3x3 with ReLU activation
7. **Flatten Layer**: Flattens the 3D output to a 1D vector
8. **Fully Connected Layer**: 64 units with ReLU activation
9. **Output Layer**: 10 units with softmax activation for classification

- **Number of Parameters**: 93,322
- **Training Accuracy**: ~99.4%
- **Test Accuracy**: ~99.3%


## Acknowledgements:

MNIST Dataset: Provided by Yann LeCun and used for training and testing the model.
Keras: A deep learning framework that simplifies the creation and training of neural networks.
TensorFlow: The underlying framework for running the Keras models.
Tkinter and Pillow: Used for creating the graphical user interface and handling image processing.

## Installation

To set up the project, you need to install the following Python libraries:

```bash
pip install tensorflow keras pillow
