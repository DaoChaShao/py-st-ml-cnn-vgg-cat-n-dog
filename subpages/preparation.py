#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/26 12:49
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preparation.py
# @Desc     :   

from streamlit import (empty, sidebar, subheader, session_state,
                       button, spinner, rerun, select_slider, number_input, slider, caption,
                       columns, image, markdown)

from utils.helper import Timer, VGG16DataProcessor
from utils.config import TRAIN_DATA_PATH, TEST_DATA_PATH

empty_messages: empty = empty()
col_train, col_test = columns(2, gap="small")

pre_sessions: list[str] = ["TRAIN", "TEST", "pTimer"]
for session in pre_sessions:
    session_state.setdefault(session, None)

with sidebar:
    subheader("Data Preparation Settings")

    if session_state["TRAIN"] is None and session_state["TEST"] is None:
        empty_messages.error("Please load data first in this section.")

        random_state: int = number_input(
            "Random State (for reproducibility)",
            0, 10_000, value=27, step=1,
            help="Set a random state for reproducibility.",
        )

        batch_size: int = select_slider(
            "Batch Size (for splitting data)",
            [16, 32, 64, 128, 256, 512], value=64,
            help="Set the batch size for splitting data into manageable chunks.",
        )

        split_rate: float = slider(
            "Train-Validation Split Rate",
            0.1, 0.5, value=0.2, step=0.1,
            help="Set the proportion of the dataset to include in the test split.",
        )

        if button("Load the Data", type="primary", width="stretch"):
            with spinner("Loading Data", show_time=True, width="stretch"):
                with Timer("Data Loading") as session_state["pTimer"]:
                    # Initialize the data processor
                    session_state["TRAIN"] = VGG16DataProcessor()
                    session_state["TEST"] = VGG16DataProcessor()
                    # Load the image data
                    session_state["TRAIN"].data_loader(TRAIN_DATA_PATH, batch_size, random_state, split_rate)
                    session_state["TEST"].data_loader(TEST_DATA_PATH, batch_size, random_state)
            rerun()
    else:
        empty_messages.success("Data loaded successfully! You can preview samples below.")

        print(type(session_state["TRAIN"]), type(session_state["TEST"]))
        print(len(session_state["TRAIN"]), len(session_state["TEST"]))

        index_batch_train = slider(
            "Select Train Batch Index to Preview",
            0, len(session_state["TRAIN"]) - 1, value=0, step=1,
            help="Select the train batch index to preview images and labels.",
        )
        index_max_train = session_state["TRAIN"][index_batch_train]
        index_item_train: int = slider(
            "Select Item Index in Train Batch to Preview",
            0, index_max_train - 1, value=0, step=1,
            help="Select the item index within the train batch to preview the image and label.",
        )

        index_batch_test = slider(
            "Select Test Batch Index to Preview",
            0, len(session_state["TEST"]) - 1, value=0, step=1,
            help="Select the test batch index to preview images and labels.",
        )
        index_max_test = session_state["TEST"][index_batch_test]
        index_item_test: int = slider(
            "Select Item Index in Test Batch to Preview",
            0, index_max_test - 1, value=0, step=1,
            help="Select the item index within the test batch to preview the image and label.",
        )

        image_train, label_train = session_state["TRAIN"][index_batch_train, index_item_train]
        image_test, label_test = session_state["TEST"][index_batch_test, index_item_test]
        caption(
            "**Note: Due to train dataset shuffling, changing the batch and item index will change the train image display.**"
        )

        print(type(image_train), type(label_train), image_train.shape, label_train.shape)
        print(type(image_test), type(label_test), image_test.shape, label_test.shape)

        with col_train:
            markdown(f"#### Train Sample at Batch {index_batch_train}, Index {index_item_train}")
            image(
                image_train.numpy().astype("uint8"),
                caption=f"Label: **{label_train.numpy()}**",
                width="stretch"
            )
        with col_test:
            markdown(f"#### Test Sample at Batch {index_batch_test}, Index {index_item_test}")
            image(
                image_test.numpy().astype("uint8"),
                caption=f"Label: **{label_test.numpy()}**",
                width="stretch"
            )
