import streamlit as st
from PIL import Image
import cv2
from ultralytics import YOLO

# Load YOLOv8 model with COCO weights
model = YOLO('yolov8n.pt')

# COCO class IDs for different vehicle types
VEHICLE_CLASSES = {
    "Car": [2],       # COCO class ID for car
    "Bike": [1, 3],   # Bicycle and motorcycle
    "Bus": [5],       # Bus
    "Truck": [7]      # Truck
}

def detect_parking_lot(image, vehicle_type):
    # Save the uploaded image temporarily
    image_path = 'temp_image.jpg'
    image.save(image_path)

    # Run YOLOv8 inference
    results = model(image_path)[0]

    # Filter detections by the selected vehicle type
    target_class_ids = VEHICLE_CLASSES.get(vehicle_type, [])
    detected_objects = [box for box in results.boxes.data if int(box[5]) in target_class_ids]

    # Determine parking lot status
    status = "Full" if len(detected_objects) > 0 else "Empty"
    annotated_image = annotate_image(image_path, detected_objects)
    return status, annotated_image

def annotate_image(image_path, detected_objects):
    # Load the original image
    img = cv2.imread(image_path)

    # Draw bounding boxes on the image
    for box in detected_objects:
        x1, y1, x2, y2, conf, cls = map(int, box[:6])
        label = f"Class {cls}: {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB for Streamlit display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# Streamlit UI
st.title("🚗 Parking Lot Detection App")
st.write("Upload an image and select the type of vehicle to detect.")

# Dropdown for selecting vehicle type
vehicle_type = st.selectbox("Select Vehicle Type", ["Car", "Bike", "Bus", "Truck"])

# Upload Image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Run detection when the button is clicked
    if st.button('Detect Parking Lot Status'):
        status, annotated_image = detect_parking_lot(image, vehicle_type)

        if status == "Empty":
            st.write(f"🎉 **Parking Lot Status: {status}!**")
            st.balloons()  # Launch balloons animation
        else:
            st.write(f"⚠️ **Parking Lot Status: {status}!**")
            st.snow()  # Launch snowflakes animation

        # Display the annotated image with detected vehicles
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)
