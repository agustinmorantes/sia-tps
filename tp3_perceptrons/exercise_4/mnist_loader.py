"""MNIST handwritten digits dataset."""

import numpy as np
import os
import urllib.request
from keras.src.utils.file_utils import get_file

def load_mnist():
    """Carga el dataset MNIST completo usando el método de Keras"""
    print("Cargando dataset MNIST...")
    
    # Usar el método de Keras para cargar MNIST
    origin_folder = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    path = get_file(
        fname="mnist.npz",
        origin=origin_folder + "mnist.npz",
        file_hash=(
            "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
        ),
    )
    
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]
    
    # Reshape a formato plano (samples, features)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    print(f"Dataset cargado:")
    print(f"  Train: {x_train.shape[0]} imágenes, {x_train.shape[1]} píxeles")
    print(f"  Test: {x_test.shape[0]} imágenes, {x_test.shape[1]} píxeles")
    print(f"  Rango de píxeles: {x_train.min()}-{x_train.max()}")
    print(f"  Clases: {np.unique(y_train)}")
    
    return x_train, y_train, x_test, y_test

def preprocess_mnist(train_images, train_labels, test_images, test_labels, normalize=True, one_hot=True):
    """Preprocesa los datos MNIST"""
    print("Preprocesando datos...")
    
    # Normalizar píxeles a [0, 1]
    if normalize:
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
    
    # One-hot encoding para las etiquetas
    if one_hot:
        train_labels_onehot = np.zeros((train_labels.shape[0], 10))
        test_labels_onehot = np.zeros((test_labels.shape[0], 10))
        
        for i in range(train_labels.shape[0]):
            train_labels_onehot[i, train_labels[i]] = 1
        
        for i in range(test_labels.shape[0]):
            test_labels_onehot[i, test_labels[i]] = 1
        
        return train_images, train_labels_onehot, test_images, test_labels_onehot, train_labels, test_labels
    
    return train_images, train_labels, test_images, test_labels

