# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load the dataset (MNIST)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalize pixel values to range 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encoding for labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the CNN model
model = Sequential()

# Add Convolutional Layer with 32 filters, 3x3 kernel, ReLU activation
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Add MaxPooling layer with 2x2 pooling size
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add Dropout to prevent overfitting
model.add(Dropout(0.25))

# Add another Convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add Dropout
model.add(Dropout(0.25))

# Flatten the data to feed it into fully connected layers
model.add(Flatten())

# Add Dense (fully connected) layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Add Dropout
model.add(Dropout(0.5))

# Output layer with 10 neurons for the 10 classes, softmax activation for multi-class classification
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Make predictions
predictions = model.predict(X_test)

# Visualize predictions
def plot_sample(X, y, predictions, index):
    plt.imshow(X[index].reshape(28, 28), cmap='gray')
    plt.title(f"True Label: {np.argmax(y[index])}, Predicted: {np.argmax(predictions[index])}")
    plt.show()

# Plot a few sample predictions
for i in range(5):
    plot_sample(X_test, y_test, predictions, i)
