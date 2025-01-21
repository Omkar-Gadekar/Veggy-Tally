import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

def detect_tomatoes(model, frame, confidence_threshold):
    results = model(frame, conf=confidence_threshold)[0]
    
    counts = {'ripe': 0, 'unripe': 0, 'defect': 0}
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = model.names[cls].lower()
        
        color_map = {
            'ripe': (0, 255, 0),     # Green
            'unripe': (0, 0, 255),   # Red
            'defect': (0, 165, 255)  # Orange
        }
        
        default_color = (255, 0, 0)  # Blue
        
        if label in counts:
            counts[label] += 1
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_map.get(label, default_color), 2)
        cv2.putText(frame, 
                    f'{label.capitalize()} {conf:.2f}', 
                    (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    color_map.get(label, default_color), 
                    2)
    
    return frame, counts

def get_working_camera_index():
    for i in range(5):  # Check indices 0 to 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

def main():
    st.set_page_config(page_title='Tomato Detection', page_icon=':tomato:', layout='wide')
    
    st.title('Tomato Detection System')
    
    try:
        model = YOLO('best_custom.pt')  # Load your YOLO model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    st.sidebar.header('Detection Settings')
    confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5)
    
    run_detection = st.checkbox('Start Detection')
    
    total_counts = {'ripe': 0, 'unripe': 0, 'defect': 0}
    
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    camera_index = get_working_camera_index()
    if camera_index is None:
        st.error("No camera found! Please connect a camera.")
        return
    
    cap = cv2.VideoCapture(camera_index)
    
    last_update_time = time.time()
    
    try:
        while run_detection:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame from the camera. Retrying...")
                time.sleep(0.1)
                continue
            
            frame = cv2.resize(frame, (640, 480))
            annotated_frame, counts = detect_tomatoes(model, frame, confidence_threshold)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            frame_placeholder.image(annotated_frame_rgb, channels='RGB')
            
            current_time = time.time()
            if current_time - last_update_time >= 10:
                for key in total_counts:
                    total_counts[key] += counts[key]
                
                stats_text = "**Detection Results:**\n"
                for label, count in total_counts.items():
                    stats_text += f"- {label.capitalize()} Tomatoes: {count}\n"
                stats_placeholder.write(stats_text)
                
                last_update_time = current_time
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
