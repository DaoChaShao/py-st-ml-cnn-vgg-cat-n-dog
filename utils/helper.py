#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/26 12:47
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :

from numpy import expand_dims
from tensorflow.data import AUTOTUNE, experimental
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from time import perf_counter
from typing import override


class Timer(object):
    """ timing code blocks using a context manager """

    def __init__(self, description: str = None, precision: int = 5):
        """ Initialise the Timer class
        :param description: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        self._description: str = description
        self._precision: int = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        """ Start the timer """
        self._start = perf_counter()
        print("-" * 50)
        print(f"{self._description} has started.")
        print("-" * 50)
        return self

    def __exit__(self, *args):
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

    def __repr__(self):
        """ Return a string representation of the timer """
        if self._elapsed != 0.0:
            # print("-" * 50)
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
        return f"{self._description} has NOT started."


class StTFKLoggerFor5Callbacks(Callback):
    """ Custom Keras Callback to log training metrics and update Streamlit placeholders.
    :param num_placeholders: a dictionary of Streamlit placeholders for metrics
    :return: None
    """

    def __init__(self, num_placeholders: dict = None):
        super().__init__()
        # The key name must match the callback logs
        self._history = {k: [] for k in [
            "loss", "accuracy", "precision", "recall", "auc",
            "val_loss", "val_accuracy", "val_precision", "val_recall", "val_auc"
        ]}
        self._placeholders = num_placeholders

    @override
    def on_epoch_end(self, epoch, logs=None):
        """ At the end of each epoch, log the metrics and update the placeholders.
        :param epoch: the current epoch number
        :param logs: the logs dictionary containing the metrics
        :return: None
        """
        logs = logs or {}
        # Save the training history per epoch
        for key in self._history.keys():
            self._history[key].append(logs.get(key, None))
        # Update the placeholders with the latest metrics
        if self._placeholders:
            for key, placeholder in self._placeholders.items():
                if key in logs and placeholder is not None:
                    placeholder.metric(
                        label=f"Epoch {epoch + 1}: {key.replace('val_', 'Valid ').capitalize()}",
                        value=f"{logs[key]:.4f}"
                    )

    def get_history(self):
        """ Get the training history."""
        return self._history


class StTFKLoggerFor2Callbacks(Callback):
    """ Custom Keras Callback to log training metrics and update Streamlit placeholders.
    :param num_placeholders: a dictionary of Streamlit placeholders for metrics
    :return: None
    """

    def __init__(self, num_placeholders: dict = None):
        super().__init__()
        # The key name must match the callback logs
        self._history = {k: [] for k in ["loss", "accuracy", "val_loss", "val_accuracy"]}
        self._placeholders = num_placeholders

    @override
    def on_epoch_end(self, epoch, logs=None):
        """ At the end of each epoch, log the metrics and update the placeholders.
        :param epoch: the current epoch number
        :param logs: the logs dictionary containing the metrics
        :return: None
        """
        logs = logs or {}
        # Save the training history per epoch
        for key in self._history.keys():
            self._history[key].append(logs.get(key, None))
        # Update the placeholders with the latest metrics
        if self._placeholders:
            for key, placeholder in self._placeholders.items():
                if key in logs and placeholder is not None:
                    placeholder.metric(
                        label=f"Epoch {epoch + 1}: {key.replace('val_', 'Valid ').capitalize()}",
                        value=f"{logs[key]:.4f}"
                    )

    def get_history(self):
        """ Get the training history."""
        return self._history


class VGG16DataProcessor(object):
    """ A class to handle data processing for VGG16 model.
    :return: None
    """

    def __init__(self):
        self._dataset = None

    def data_loader(self, data_path: str, batch_size: int, seed: int, split_rate: float | None = 0.2):
        """ Load image data from a directory.
        :param data_path: the path to the image directory
        :param batch_size: the size of each data batch
        :param seed: the random seed for shuffling
        :param split_rate: the proportion of the dataset to include in the validation split
        :return: None
        """
        if "train" in data_path:
            shuffle: bool = True
            val_split: float | None = split_rate
            subset: str | None = "training"
        else:
            shuffle: bool = False
            val_split: float | None = None
            subset: str | None = None

        self._dataset = image_dataset_from_directory(
            data_path,
            labels="inferred",
            label_mode="int",
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            validation_split=val_split,
            subset=subset,
            interpolation="bilinear",
        )

    def data_normalizer(self):
        """ Normalize the image data using VGG16 preprocessing.
        :return: None
        """
        if self._dataset is not None:
            self._dataset = self._dataset.map(lambda x, y: (preprocess_input(x), y))
            self._dataset = self._dataset.prefetch(AUTOTUNE)

    def __len__(self) -> int:
        if self._dataset is not None:
            return experimental.cardinality(self._dataset).numpy()
        return 0

    def __getitem__(self, key):
        if self._dataset is None:
            raise ValueError("Dataset is None")

        if isinstance(key, int):
            batch = key
            for b, (images, labels) in enumerate(self._dataset):
                if b == batch:
                    return images.shape[0]
            raise IndexError(f"Batch index {batch} out of range")

        elif isinstance(key, tuple) and len(key) == 2:
            batch, index = key
            print(batch, index)
            for b, (images, labels) in enumerate(self._dataset):
                if b == batch:
                    return images[index], labels[index]
            raise IndexError(f"Index {index} out of range")
        else:
            raise TypeError("Invalid key type. Must be int or tuple of two ints.")

    def __repr__(self) -> str:
        if self._dataset is not None:
            for images, labels in self._dataset.take(1):
                return f"Dataset: Image Shape in First Batch={images.shape}, Labels Shape in First Batch={labels.shape}"
        return "Dataset is None"

    def getter(self):
        if self._dataset is not None:
            return self._dataset
        raise ValueError("Dataset is None")


def vgg16_data_augmenter() -> Sequential:
    """ Create a data augmentation pipeline for VGG16 model.
    :return: A Sequential model containing data augmentation layers.
    """
    return Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomHue(0.1),
    ])


def single_data_loader(img_path: str, img_height: int = 224, img_width: int = 224, normalise: bool = False):
    # Load and resize the image
    img = load_img(img_path, target_size=(img_height, img_width))
    img_arr = img_to_array(img)
    # Create a batch dimension
    img_batch = expand_dims(img_arr, axis=0).astype("float32")

    if normalise:
        # Normalise the image data
        img_preprocessed = preprocess_input(img_batch)
    else:
        img_preprocessed = img_batch
    print(type(img_preprocessed), img_preprocessed.shape)

    return img_preprocessed
