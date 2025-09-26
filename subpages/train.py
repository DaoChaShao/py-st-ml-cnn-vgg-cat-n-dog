#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/26 12:49
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   train.py
# @Desc     :   

from os import path, remove
from tensorflow.keras.applications.vgg16 import VGG16
from streamlit import (empty, sidebar, subheader, session_state,
                       button, spinner, rerun, number_input,
                       columns)
from tensorflow.keras import models, layers, metrics

from utils.config import MODEL_SAVE_PATH
from utils.helper import Timer, vgg16_data_augmenter, StTFKLoggerFor5Callbacks

empty_messages: empty = empty()
empty_result_title: empty = empty()
col_los, col_acc, col_pre, col_rec, col_auc = columns(5, gap="small")
col_los_valid, col_acc_valid, col_pre_valid, col_rec_valid, col_auc_valid = columns(5, gap="small")
placeholder_los = col_los.empty()
placeholder_acc = col_acc.empty()
placeholder_pre = col_pre.empty()
placeholder_rec = col_rec.empty()
placeholder_auc = col_auc.empty()
placeholder_los_val = col_los_valid.empty()
placeholder_acc_val = col_acc_valid.empty()
placeholder_pre_val = col_pre_valid.empty()
placeholder_rec_val = col_rec_valid.empty()
placeholder_auc_val = col_auc_valid.empty()

pre_sessions: list[str] = ["TRAIN", "TEST"]
for session in pre_sessions:
    session_state.setdefault(session, None)
model_sessions: list[str] = ["model", "histories", "mTimer"]
for session in model_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if session_state["TRAIN"] is None and session_state["TEST"] is None:
        empty_messages.error("Please upload training and testing data first in the **Preparation Page** page.")
    else:
        empty_messages.info("You can train the model in this section.")
        subheader("Model Training Settings")

        # Initialize the metrics placeholders
        placeholders: dict = {
            "loss": placeholder_los,
            "accuracy": placeholder_acc,
            "precision": placeholder_pre,
            "recall": placeholder_rec,
            "auc": placeholder_auc,
            "val_loss": placeholder_los_val,
            "val_accuracy": placeholder_acc_val,
            "val_precision": placeholder_pre_val,
            "val_recall": placeholder_rec_val,
            "val_auc": placeholder_auc_val
        }
        # Initialise the callback for visualisation
        callback = StTFKLoggerFor5Callbacks(placeholders)

        # Normalise the data
        # session_state["TRAIN"].data_normalizer()

        if session_state["model"] is None:
            epochs: int = number_input(
                "Epochs (number of training iterations)",
                1, 100, value=3, step=1,
                help="Set the number of epochs for training the model.",
            )

            if button("Train the VGG16 Model", type="primary", width="stretch"):
                with spinner("Training Model", show_time=True, width="stretch"):
                    with Timer("Model Training") as session_state["mTimer"]:
                        # Data Augmentation
                        augmentation = vgg16_data_augmenter()
                        # Set the vgg16 model
                        model = VGG16(
                            weights="imagenet",  # Download weights pre-trained on ImageNet while running the code
                            input_shape=(224, 224, 3),
                            include_top=False,
                            pooling="max"
                        )
                        # Freeze the conv layers
                        model.trainable = False
                        # Train the model
                        session_state["model"] = models.Sequential([
                            augmentation,
                            model,
                            layers.Flatten(),
                            layers.Dense(128, activation="relu"),
                            layers.Dropout(0.3),
                            layers.Dense(1, activation="sigmoid"),
                        ])
                        print(session_state["model"].summary())

                        session_state["model"].compile(
                            optimizer="adam",
                            loss="binary_crossentropy",
                            metrics=[
                                "accuracy",
                                metrics.Precision(name="precision"),
                                metrics.Recall(name="recall"),
                                metrics.AUC(name="auc"),
                            ],
                        )

                        session_state["model"].fit(
                            session_state["TRAIN"].getter(),
                            validation_data=session_state["TEST"].getter(),
                            epochs=epochs,
                            callbacks=[callback],
                        )
                        session_state["histories"] = callback.get_history()
                rerun()
        else:
            empty_result_title.markdown("### Model Training Result")
            hist = session_state["histories"]
            if hist:
                last_epoch = len(hist["loss"])
                for key, placeholder in placeholders.items():
                    if key in hist and placeholder is not None:
                        value = hist[key][-1]
                        label = f"Epoch {last_epoch}: {key.replace("val_", "Valid ").capitalize()}"
                        placeholder.metric(label=label, value=f"{value:.4f}")

            if path.exists(MODEL_SAVE_PATH):
                empty_messages.info(
                    f"The model file **{MODEL_SAVE_PATH}** already exists in the current directory."
                )

                if button("Delete the Model", type="secondary", width="stretch"):
                    with spinner("Deleting the model...", show_time=True, width="stretch"):
                        with Timer("Deleting the model") as timer:
                            remove(MODEL_SAVE_PATH)

                            for placeholder in placeholders.values():
                                placeholder.empty()

                            for session in model_sessions:
                                session_state[session] = None
                    empty_messages.success(f"{timer} The model has been deleted successfully!")
                    rerun()
            else:
                empty_messages.success(
                    f"{session_state["mTimer"]} The model has been trained successfully! You can save the model."
                )

                if button("Save the Model", type="primary", width="stretch"):
                    with spinner("Saving the model...", show_time=True, width="stretch"):
                        with Timer("Saving the model") as timer:
                            session_state["model"].save(MODEL_SAVE_PATH)
                    empty_messages.success(f"{timer} The model has been saved successfully!")
                    rerun()
