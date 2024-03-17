#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import roboflow as rf
from roboflow import Roboflow
import supervision as sv
import tempfile
import cv2
import numpy as np
import os

# Initialize Roboflow with your API key
rf = Roboflow(api_key="xJFjzUFI5Aj9scn14Aki")
project = rf.workspace().project("0702241217")
model = project.version(1).model

st.title("Shrimp Detection and Analysis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
 if image is not None:
        # Resize the image to reduce its dimensions, if necessary
        # Define the desired width and height or scale factor
        desired_width = 1024
        scale_factor = desired_width / image.shape[1]
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.flush()  # Make sure data is written to disk
        tmp_file_path = tmp_file.name

    # Read the file using OpenCV
    image = cv2.imread(tmp_file_path)
    
    # Check if the image was successfully loaded
    if image is not None:
        # Display the uploaded image
        st.image(image, channels="BGR", caption="Uploaded Image")

        # Make prediction on the uploaded image using the path of the temporary file
        result = model.predict(tmp_file_path, confidence=30).json()

        # Adaptation to the updated API
        detections = sv.Detections.from_inference(result)

        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        annotated_image = mask_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # Display the annotated image
        st.image(annotated_image, channels="BGR", caption="Annotated Image")

         # Initialize a list to store the number of pixels for each shrimp
        shrimp_areas = []
       
        # Directly iterate over the xyxy bounding box coordinates
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)

            # Crop the region of interest (ROI) from the image
            roi = image[y1:y2, x1:x2]

            # Convert the ROI to grayscale
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Count non-zero pixels (representing the shrimp region)
            area = cv2.countNonZero(roi_gray)

            # Append the area to the list
            shrimp_areas.append(area)

        # Sort the shrimp areas in descending order
        shrimp_areas.sort(reverse=True)
        st.write("shrimp_areas",shrimp_areas)

        # Display shrimp areas and calculate ratios (simplified for brevity)
        if shrimp_areas:
            top_10_ratio = sum([(area ** 1.5) for area in shrimp_areas[:10]])
            bottom_10_ratio = sum([(area ** 1.5) for area in shrimp_areas[-10:]])
            overall_ratio = top_10_ratio / bottom_10_ratio

            st.write("Top 10 shrimp areas^1.5:", top_10_ratio)
            st.write("Bottom 10 shrimp areas^1.5:", bottom_10_ratio)
            st.write("Overall Ratio:", overall_ratio)

        # Cleanup: remove the temporary file
        os.remove(tmp_file_path)
    else:
        st.error("Failed to read the image. Please check the file format and try again.")
        # Cleanup even if the image fails to load
        os.remove(tmp_file_path)

