#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import roboflow as rf
from roboflow import Roboflow
import supervision as sv
import tempfile
import cv2
import numpy as np
import os
import math

# Initialize Roboflow with your API key
rf = Roboflow(api_key="CFNTjMGFICYz3vLozs3A")
project = rf.workspace().project("prawn-qoy6a")
model = project.version(2).model

st.title("Shrimp Detection and Analysis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.flush()  # Make sure data is written to disk
        tmp_file_path = tmp_file.name

    # Read the file using OpenCV
    image = cv2.imread(tmp_file_path)
    
    # Check if the image was successfully loaded
    if image is not None:
        desired_width = 1024
        scale_factor = desired_width / image.shape[1]
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        # Display the uploaded image
        st.image(image, channels="BGR", caption="Uploaded Image")

        # Make prediction on the uploaded image using the path of the temporary file
        result = model.predict(tmp_file_path, confidence=50).json()

        # Adaptation to the updated API
        detections = sv.Detections.from_inference(result)

        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        annotated_image = mask_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # Initialize a list to store the number of pixels for each shrimp
        shrimp_areas = []

        # Directly iterate over the xyxy bounding box coordinates
        for idx, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox)

            # Crop the region of interest (ROI) from the segmented image
            roi_segmented = annotated_image[y1:y2, x1:x2]

            # Calculate text position for the annotation
            text_position = (x1, y1 - 10)  # Adjust text position as needed

            # Annotate the segmented image with shrimp number
            cv2.putText(annotated_image, str(idx + 1), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            # Convert the ROI to grayscale
            roi_gray = cv2.cvtColor(roi_segmented, cv2.COLOR_BGR2GRAY)

            # Count non-zero pixels (representing the shrimp region)
            area = cv2.countNonZero(roi_gray)

            # Append the area to the list
            shrimp_areas.append(area)

        # Display the annotated segmented image
        st.image(annotated_image, channels="BGR", caption="Annotated Segmented Image")

        # Sort the shrimp areas in descending order
        shrimp_areas.sort(reverse=True)
        if shrimp_areas:
            st.write("Shrimp areas (in pixels):")
            for idx, area in enumerate(shrimp_areas):
                st.write(f"{idx + 1}: {area}")

        # Prompt the user to input shrimp per pound
        shrimp_per_lb = st.number_input("Enter the shrimp per pound:", value=30, step=1)

        # Display shrimp areas and calculate ratios (simplified for brevity)
        shrimp_areas.sort(reverse=False)
        if shrimp_areas:
            num_shrimp = len(shrimp_areas)
  
            if 4 <= shrimp_per_lb <= 20:
                top_count = min(num_shrimp, 5)
                bottom_count = min(num_shrimp, 5)

            elif 21 <= shrimp_per_lb <= 60:
                top_count = min(num_shrimp, 10)
                bottom_count = min(num_shrimp, 10)

            elif 61 <= shrimp_per_lb <= 200:
                top_count = min(num_shrimp, 20)
                bottom_count = min(num_shrimp, 20)

            elif 201 <= shrimp_per_lb <= 500:
                top_count = min(num_shrimp, 25)
                bottom_count = min(num_shrimp, 25)

            top_10_percent_index_start = num_shrimp - top_count
            top_10_percent_index_end = num_shrimp  # Include all items in top count

            bottom_10_percent_index_end = bottom_count  # Include all items in bottom count

            sum_top_10_percent_areas = sum([area ** .676 for area in shrimp_areas[top_10_percent_index_start:top_10_percent_index_end]])
            sum_bottom_10_percent_areas = sum([area ** .676 for area in shrimp_areas[:bottom_10_percent_index_end]])

            overall_ratio = sum_top_10_percent_areas / sum_bottom_10_percent_areas if sum_bottom_10_percent_areas != 0 else float('inf')

            st.write("Sum of top", top_count, "shrimp areas:", sum_top_10_percent_areas)
            st.write("Sum of bottom", bottom_count, "shrimp areas:", sum_bottom_10_percent_areas)
            st.write("Overall Ratio:", overall_ratio)

            # Cleanup: remove the temporary file
            os.remove(tmp_file_path)
    else:
        st.error("Failed to read the image. Please check the file format and try again.")
        # Cleanup even if the image fails to load
        os.remove(tmp_file_path)
