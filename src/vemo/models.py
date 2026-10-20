"""Models compared in the experiments.

All expose ``fit(X, y)`` and ``predict(X)`` on string labels, so one experiment
loop can score every one of them under the same split.
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def make_majority() -> DummyClassifier:
    """Always predicts the most frequent class. Any real model has to beat this."""
    return DummyClassifier(strategy="most_frequent")


def make_linear(c: float = 1.0) -> Pipeline:
    """Standardised features into a class-balanced logistic regression."""
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=c, max_iter=5000, class_weight="balanced")
    )


class OriginalCNN:
    """The CNN from the first version of this project, layer for layer.

    Kept unchanged on purpose: the point of the comparison is that the same model
    scores very differently once the evaluation stops leaking.
    """

    def __init__(self, epochs: int = 50, batch_size: int = 32, seed: int = 0, verbose: int = 0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.encoder = LabelEncoder()
        self.model = None

    def _build(self, input_shape: tuple[int, ...], n_classes: int):
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import (
            BatchNormalization,
            Conv2D,
            Dense,
            Dropout,
            Flatten,
            Input,
            MaxPooling2D,
        )

        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(32, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                BatchNormalization(),
                Dropout(0.25),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D((2, 2)),
                BatchNormalization(),
                Dropout(0.25),
                Flatten(),
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(n_classes, activation="softmax"),
            ]
        )
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "OriginalCNN":
        """``X_val`` drives the learning-rate schedule only.

        The original passed the test set here. The honest protocol passes held-out
        training actors instead, so the test set never influences training.
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import ReduceLROnPlateau

        tf.keras.utils.set_random_seed(self.seed)
        self.encoder.fit(np.concatenate([y, y_val]))
        self.model = self._build(X.shape[1:] + (1,), len(self.encoder.classes_))
        self.model.fit(
            X[..., None],
            self.encoder.transform(y),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val[..., None], self.encoder.transform(y_val)),
            callbacks=[ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-4)],
            verbose=self.verbose,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict(X[..., None], verbose=0)
        return self.encoder.inverse_transform(probs.argmax(axis=1))
