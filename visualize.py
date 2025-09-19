import NeuralNet as NN
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

pixels = 64

nn1 = NN.NeuralNetModel()

nn1.load_model('DLNNmodel.pkl')

def load(parentDir, image_size=(pixels, pixels)):
    images = []
    labels = []

    subdirs = [(os.path.join(parentDir, d), d) for d in sorted(os.listdir(parentDir)) if os.path.isdir(os.path.join(parentDir, d))]
    for subdir_path, label in subdirs:
        for filename in os.listdir(subdir_path):
            if (filename.endswith('.png') or filename.endswith('.jpg')):
                img_path = os.path.join(subdir_path, filename)
                with Image.open(img_path) as img:
                    img = img.convert('L').resize(image_size)
                    images.append(np.array(img).reshape(-1))
                labels.append(0 if label == 'h' else 1)
    return np.array(images), np.array(labels).reshape(-1, 1)

# Load the data
images, labels = load('dataset')

# Reshape the data
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

nn1.visualize_network(X_test, is_double=True)