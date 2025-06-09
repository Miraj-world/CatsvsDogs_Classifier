# 🐱🐶 Cats vs Dogs Image Classifier (TensorFlow/Keras)

This project is a Convolutional Neural Network (CNN)-based image classifier that distinguishes between **cats** and **dogs** using TensorFlow and Keras. Achieved **87% accuracy** on the validation set!

## 📁 Project Structure

cats-vs-dogs-cnn/
├── cats_vs_dogs_CNN_87accuracy.ipynb # Main Jupyter notebook
├── model/ # (Optional) Saved trained model
├── README.md # You're here

## 📊 Dataset

Used the **TensorFlow Datasets** version of the `cats_vs_dogs` dataset:

- 23262 total images
- 2 classes: **Cats (label=0)**, **Dogs (label=1)**
- Images were resized for faster training (e.g., 64x64 or 128x128)

```python
import tensorflow_datasets as tfds
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

🧠 Model Architecture
Built using tf.keras.Sequential with a CNN architecture like:
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

Loss Function: Binary Crossentropy

Optimizer: Adam

Metric: Accuracy

🧪 Results

- ✅ Validation Accuracy: ~87%

- 🧪 Includes random image prediction with model output vs true label

- 📉 Includes training/validation loss and accuracy plots



🧪 Sample Prediction Output

| Image                            | Prediction   | Actual |
| -------------------------------- | ------------ | ------ |
| ![sample](images/sample_dog.png) | `Dog` (0.92) | `Dog`  |
| ![sample](images/sample_cat.png) | `Cat` (0.88) | `Cat`  |

💾 Saving & Loading the Model

# Save
model.save('model/cats_vs_dogs_cnn.h5')

# Load
from tensorflow.keras.models import load_model
model = load_model('model/cats_vs_dogs_cnn.h5')

🚀 Future Improvements

- Add data augmentation (rotation, flipping)

- Use transfer learning with pre-trained models like MobileNet or VGG16

- Deploy using Streamlit or Flask

📌 Requirements

Install these before running:
	pip install tensorflow matplotlib numpy

💡 Credits

Dataset: TensorFlow Datasets - cats_vs_dogs

Frameworks: TensorFlow, Keras, NumPy, Matplotlib

📎 License
MIT License. Free to use and modify.



