#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/26 12:48
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :

from streamlit import title, expander, caption, empty

empty_messages = empty()
empty_messages.info("Please check the details at the different pages of core functions.")

title("Convolutional Neural Network (CNN) - VGG16 for Cat & Dog Classification")
with expander("**INTRODUCTION**", expanded=True):
    caption("This Streamlit app provides a complete workflow for training, testing, and visualizing a Convolutional Neural Network (CNN) for binary image classification (Cats vs Dogs).")
    caption("+ **Data Preparation**: Load, preview, and explore your training and testing image datasets.")
    caption("+ **Model Training**: Configure settings, apply data augmentation, and train the VGG16 model with real-time metric tracking.")
    caption("+ **Model Testing**: Evaluate model performance on the test dataset and preview predictions for individual samples.")
    caption("+ **Realtime Prediction**: Upload any image and get instant predictions using the trained VGG16 model.")
