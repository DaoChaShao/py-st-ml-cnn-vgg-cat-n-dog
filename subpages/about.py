#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/26 12:48
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   about.py
# @Desc     :   

from streamlit import title, expander, caption

title("**Application Information**")
with expander("About this application", expanded=True):
    caption("This application demonstrates end-to-end deep learning workflow using TensorFlow and Streamlit:")
    caption("- **Data Preparation Page:** Allows users to load datasets and set batch size, random seed, and validation split.")
    caption("- **Training Page:** Users can train the VGG16-based model and monitor metrics like loss, accuracy, precision, recall, and AUC in real-time.")
    caption("- **Testing Page:** Evaluate the model on test data, display metrics, and preview images with predicted labels.")
    caption("- **Realtime Prediction Page:** Upload a single image and obtain immediate prediction results with visual feedback.")
    caption("- **Streamlit Integration:** All pages are interactive and provide instant feedback using Streamlit placeholders and widgets.")
