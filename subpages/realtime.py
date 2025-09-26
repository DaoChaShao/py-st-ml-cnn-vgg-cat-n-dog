#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/26 12:49
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   realtime.py
# @Desc     :   

from os import path, remove
from streamlit import (empty, sidebar, subheader, session_state,
                       button, spinner, rerun, file_uploader,
                       columns, markdown, image)
from tensorflow.keras.models import load_model

from utils.config import MODEL_SAVE_PATH
from utils.helper import Timer, single_data_loader

empty_messages: empty = empty()
empty_samp_title: empty = empty()
col_img, col_num = columns(2, gap="small")

pre_sessions: list[str] = ["TRAIN", "TEST"]
for session in pre_sessions:
    session_state.setdefault(session, None)
pred_sessions: list[str] = ["pdTimer", "img", "pred"]
for session in pred_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if session_state["TRAIN"] is None and session_state["TEST"] is None:
        empty_messages.error("Please upload training and testing data first in the **Preparation Page** page.")
    else:
        if not path.exists(MODEL_SAVE_PATH):
            empty_messages.error("Please train the model first in the **Training Page** page.")
        else:
            subheader("Prediction Settings")

            uploaded_file = file_uploader(
                "Upload an image for prediction",
                type=["png", "jpg", "jpeg"],
                help="Upload an image file in PNG, JPG, or JPEG format for real-time prediction.",
            )

            if uploaded_file is None:
                empty_messages.warning("Please upload an image file to proceed with prediction.")
            else:
                empty_messages.info("You can get real-time prediction in this section.")

                # Load the trained model
                model = load_model(MODEL_SAVE_PATH)

                if session_state["img"] is None and session_state["pred"] is None:
                    if button("Predict the Image", type="primary", width="stretch"):
                        with spinner("Predicting the Image", show_time=True, width="stretch"):
                            with Timer("Real-time Prediction") as session_state["pdTimer"]:
                                # Preprocess the uploaded image
                                session_state["img"] = single_data_loader(uploaded_file)
                                if session_state["img"] is None:
                                    empty_messages.error(
                                        "Failed to load the image. Please ensure the file is a valid image."
                                    )
                                else:
                                    # Make prediction
                                    pred_prob = model.predict(session_state["img"])
                                    session_state["pred"] = (pred_prob > 0.5).astype("int32").flatten()[0]
                        rerun()
                else:
                    empty_messages.success(
                        f"{session_state['pdTimer']} Image predicted successfully! You can click the **Reset Prediction** button and **Reupload** the image to repredict."
                    )

                    with col_img:
                        empty_samp_title.markdown("### Test Sample Display Area")
                        image(
                            session_state["img"].astype("uint8"),
                            caption=(
                                f"True Label: **{'cat' if session_state["pred"] == 0 else 'dog'}**"
                            ),
                            width="stretch"
                        )
                    with col_num:
                        markdown(
                            f"<h1 style='font-size:240px; font-weight:bold; text-align:center;'>{'cat' if session_state["pred"] == 0 else 'dog'}</h1>",
                            unsafe_allow_html=True, width="stretch"
                        )

                    if button("Reset Prediction", type="secondary", width="stretch"):
                        for session in pred_sessions:
                            session_state[session] = None
                        # Clean up the uploaded file
                        uploaded_file = None
                        rerun()
