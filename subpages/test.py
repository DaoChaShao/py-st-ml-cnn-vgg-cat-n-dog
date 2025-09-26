#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/26 12:49
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   test.py
# @Desc     :   

from numpy import concatenate
from os import path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from streamlit import (empty, sidebar, subheader, session_state,
                       button, spinner, rerun, caption, slider,
                       columns, image, markdown)
from tensorflow.keras.models import load_model

from utils.config import MODEL_SAVE_PATH
from utils.helper import Timer

empty_messages: empty = empty()
empty_samp_title: empty = empty()
col_img, col_num = columns(2, gap="small")
empty_result_title: empty = empty()
col_acc, col_pre, col_rec, col_auc, col_f1 = columns(5, gap="small")

pre_sessions: list[str] = ["TRAIN", "TEST"]
for session in pre_sessions:
    session_state.setdefault(session, None)
model_sessions: list[str] = ["model"]
for session in model_sessions:
    session_state.setdefault(session, None)
test_sessions: list[str] = ["tTimer", "y_pred"]
for session in test_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if session_state["TRAIN"] is None and session_state["TEST"] is None:
        empty_messages.error("Please upload training and testing data first in the **Preparation Page** page.")
    else:
        if not path.exists(MODEL_SAVE_PATH):
            empty_messages.error("Please train the model first in the **Training Page** page.")
        else:
            subheader("Model Testing Settings")

            # Load the trained model
            model = load_model(MODEL_SAVE_PATH)
            # Normalise the data
            # session_state["TEST"].data_normalizer()

            if session_state["y_pred"] is None:
                empty_messages.info("You can test the model in this section.")

                if button("Test the Model", type="primary", width="stretch"):
                    with spinner("Testing the Model", show_time=True, width="stretch"):
                        with Timer("Model Testing") as session_state["tTimer"]:
                            # Evaluate the model on the test data
                            y_pred_prob = model.predict(session_state["TEST"].getter(), verbose=1)
                            print(y_pred_prob)
                            session_state["y_pred"] = (y_pred_prob > 0.5).astype("int32").flatten()
                    rerun()
            else:
                empty_messages.success(
                    f"{session_state["tTimer"]} Model tested successfully! You can view the results below."
                )

                print(session_state["y_pred"])

                y_true = concatenate([y for _, y in session_state["TEST"].getter()], axis=0)

                empty_result_title.markdown("#### Test Results")
                accuracy = accuracy_score(y_true, session_state["y_pred"])
                precision = precision_score(y_true, session_state["y_pred"])
                recall = recall_score(y_true, session_state["y_pred"])
                auc = roc_auc_score(y_true, session_state["y_pred"])
                f1 = f1_score(y_true, session_state["y_pred"])
                with col_acc:
                    col_acc.metric("Accuracy", f"{accuracy:.3%}")
                with col_pre:
                    col_pre.metric("Precision", f"{precision:.3%}")
                with col_rec:
                    col_rec.metric("Recall", f"{recall:.3%}")
                with col_auc:
                    col_auc.metric("AUC", f"{auc:.4f}")
                with col_f1:
                    col_f1.metric("F1-Score", f"{f1:.4f}")

                amount_batch: int = len(session_state["TEST"]) - 1
                index_batch_test = slider(
                    "Select Test Batch Index to Preview",
                    0, amount_batch, value=0, step=1,
                    help="Select the test batch index to preview images and labels.",
                )
                caption(f"Note: the test values are in [0, {amount_batch}].")
                index_max_test = session_state["TEST"][index_batch_test]
                amount_test: int = index_max_test - 1
                index_item_test: int = slider(
                    "Select Item Index in Test Batch to Preview",
                    0, amount_test, value=0, step=1,
                    help="Select the item index within the test batch to preview the image and label.",
                )
                caption(f"Note: the index is in [0, {amount_test}].")
                image_test, label_test = session_state["TEST"][index_batch_test, index_item_test]
                caption(f"Note: the image named **1.dog.0** in tester directory will surprise you.")

                if button("Predict the Selected Sample", type="primary", width="stretch"):
                    with spinner("Predicting the selected sample", show_time=True, width="stretch"):
                        with Timer("Sample Prediction") as timer:
                            with col_img:
                                empty_samp_title.markdown(
                                    f"### Test Sample at Index {index_item_test} of epoch {index_batch_test}"
                                )
                                image(
                                    image_test.numpy().astype("uint8"),
                                    caption=(
                                        f"True Label: **{'cat' if label_test.numpy() == 0 else 'dog'}**"
                                    ),
                                    width="stretch"
                                )
                            with col_num:
                                print(type(image_test.numpy().astype("uint8")))
                                print(image_test.numpy().astype("uint8").shape)
                                single_image_for_test = image_test.numpy().astype("uint8").reshape(
                                    1,
                                    *image_test.numpy().astype("uint8").shape
                                )
                                pred_prob = model.predict(single_image_for_test)
                                pred_label = (pred_prob > 0.5).astype("int32").flatten()[0]
                                print(pred_label)

                                markdown(
                                    f"<h1 style='font-size:240px; font-weight:bold; text-align:center;'>{'cat' if pred_label == 0 else 'dog'}</h1>",
                                    unsafe_allow_html=True, width="stretch"
                                )
