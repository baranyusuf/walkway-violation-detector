from ultralytics import YOLO
import cv2
from sort import Sort
import numpy as np
import logging
import time
from datetime import datetime
import os
import csv
import threading


# ----------------------- Firebase Initialization -----------------------
import firebase_admin
from firebase_admin import credentials, db, storage


# Replace this JSON file name with the path to your actual service account key file.
firebase_cred = credentials.Certificate("")


# IMPORTANT: Use your actual Realtime Database URL and storageBucket here.
firebase_admin.initialize_app(firebase_cred, {
    'databaseURL': '',
    'storageBucket': ''  # Replace with your Firebase bucket
})

tracked_people_lock = threading.Lock()

def upload_to_firebase(local_path, track_id, tracked_people):
    bucket = storage.bucket()
    blob = bucket.blob(f"violations/trackID_{track_id}_{int(time.time())}.jpg")

    # Asynchronously read and upload the file content
    with open(local_path, 'rb') as f:
        data = f.read()
        blob.upload_from_string(data, content_type='image/jpeg')  # Send in-memory
        blob.make_public()
    
    with tracked_people_lock:
        if track_id in tracked_people:
            tracked_people[track_id]["image_url"] = blob.public_url


def send_violation_info_to_database(violation_data):
    # Save the violation event under /violations
    db.reference("violations").push(violation_data)



# -----------------------------------------------------------------------

# Initialize CSV file for HSV values
# hsv_csv_file = "hsv_values_high.csv"
# with open(hsv_csv_file, mode='w', newline='') as hsv_file:
#     hsv_writer = csv.writer(hsv_file)

# Initialize CSV file
total_positives_csv = "total_positive_detections.csv"
with open(total_positives_csv, mode='w', newline='') as file:
    csv_writer = csv.writer(file)

# Initialize CSV file
tracking_durations_csv = "tracking_durations.csv"
with open(tracking_durations_csv, mode='w', newline='') as file:
    csv_writer = csv.writer(file)

logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Video file or webcam source
VIDEO_FILE = "Videos/TestVideo1.mp4"

# Directory to save videos of tracked people
output_video_dir = "Violations"
if not os.path.exists(output_video_dir):
    os.makedirs(output_video_dir)


stream = ''

# before opening:
#os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


# Initialize video capture
cap = cv2.VideoCapture(stream)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
resize_width = frame_width
resize_height = int((resize_width / frame_width) * frame_height) if frame_width > 0 else 480
print("Actual Frame Width x Height: ", frame_width, "x", frame_height)
print("Resized Frame Width x Height: ", resize_width, "x", resize_height)

# Initialize YOLO and SORT
chosen_model = YOLO("yolo11l.pt").to("cuda")  # YOLO model
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)  # SORT tracker

# Variables for mouse selection
area = []
is_drawing = False
finalized = False

# Mouse callback function
def select_area(event, x, y, flags, param):
    global area, is_drawing, finalized
    if event == cv2.EVENT_LBUTTONDOWN:
        if not finalized:
            is_drawing = True
            area.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing and not finalized:
            area[-1] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if is_drawing and not finalized:
            is_drawing = False
            area.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right-click to finalize the selection
        if len(area) >= 3:
            finalized = True

# Set up mouse callback
cv2.namedWindow("Select Area", cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("Select Area", select_area)

# Read the first frame for area selection
success, frame = cap.read()
if success:
    frame = cv2.resize(frame, (resize_width, resize_height))
    while True:
        # Draw the area on the frame
        if len(area) > 0:
            cv2.polylines(frame, [np.array(area, np.int32)], isClosed=finalized, color=(255, 0, 255), thickness=2)
            for point in area:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.imshow("Select Area", frame)
        if finalized:
            key = cv2.waitKey(0)  # Wait for Enter to proceed
            if key == 13:  # Enter key
                break
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            cap.release()
            cv2.destroyAllWindows()
            exit()
    # Convert the area to NumPy array
    walkway_area = np.array(area, np.int32)
else:
    print("Error reading the video.")
    exit()

hsv_values = []
# Convert frame to HSV
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Create a mask for the polygon
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [walkway_area], 255)
for y in range(hsv_frame.shape[0]):
    for x in range(hsv_frame.shape[1]):
        if mask[y, x] == 255:
            hsv_pixel = hsv_frame[y, x]
            hsv_values.append(hsv_pixel)

# Restart the video from the beginning
cap.release()
cap = cv2.VideoCapture(stream)

tracked_people = {}  # Dictionary to track detected persons (violations)
track_frame_width = 200
track_frame_height = 400



frame_count = 0
frame_start_time = time.time()

fps = 0

# Main loop for detection and tracking
while True:
    success, frame = cap.read()
    

    if not success:
        break

    frame_count += 1


    if (time.time() - frame_start_time) >= 1:
        elapsed = time.time() - frame_start_time
        fps = frame_count / elapsed
        # print(f"Effective Input FPS: {fps:.2f}")
        frame_count = 0
        frame_start_time = time.time()



    # Resize frame
    frame = cv2.resize(frame, (resize_width, resize_height))

    # Runs YOLO to detect pedestrians (class 0) with a confidence threshold of 0.5
    results = chosen_model.predict(frame, classes=[0], conf=0.25, device="cuda")
    violation_detections = []
    total_people = []

    # Draw the walkway area
    cv2.polylines(frame, [walkway_area], True, (0, 255, 255), 2)

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            # Check if the bottom-center of the bounding box is within the walkway area
            in_walkway = cv2.pointPolygonTest(walkway_area, (int((int(x_min) + int(x_max)) / 2), int(y_max)), False)
            total_people.append(box)
            if in_walkway >= 0:
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            else:
                violation_detections.append([x_min, y_min, x_max, y_max, conf])

    # Write only the number of violations to the CSV
    with open(total_positives_csv, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([len(total_people)])

    # Convert to NumPy array (required by SORT)
    violation_detections = np.array(violation_detections)
    if len(violation_detections) > 0:
        tracked_objects = tracker.update(violation_detections)
    else:
        tracked_objects = []

    people_outside_walkway = [] 

    # Draw tracking results on the frame
    for obj in tracked_objects:
        x_min, y_min, x_max, y_max, track_id = obj
        track_id = int(track_id)
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frame, (int((int(x_min)+int(x_max))/2), int(y_max)), 4, (255,0,0), -1)

        people_outside_walkway.append(track_id)

        # Expand bounding box by 50 pixels in all directions
        x_min_exp = max(0, int(x_min) - 50)
        y_min_exp = max(0, int(y_min) - 50)
        x_max_exp = min(resize_width, int(x_max) + 50)
        y_max_exp = min(resize_height, int(y_max) + 50)

        cropped_frame = frame[y_min_exp:y_max_exp, x_min_exp:x_max_exp]
        cropped_frame = cv2.resize(cropped_frame, (track_frame_width, track_frame_height))
        
        if track_id not in tracked_people:
            tracked_people[track_id] = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                "missed_frames": 0
            }
            output_filename = f"{output_video_dir}/trackID_{track_id}_violation.jpg"
            cv2.imwrite(output_filename, cropped_frame)

            # Start a new thread for uploading the image
            upload_thread = threading.Thread(target=upload_to_firebase, args=(output_filename, track_id, tracked_people))
            upload_thread.start()

            print(f"Walkway Violation Detected --> Person ID: {track_id} at {tracked_people[track_id]['start_time']}")
        else:
            # Reset missed frames if we see them again
            tracked_people[track_id]["missed_frames"] = 0


    # Identify tracks to remove
    tracks_to_remove = [] 
    for track_id in list(tracked_people.keys()):
        if track_id not in people_outside_walkway:
            tracked_people[track_id]["missed_frames"] += 1
            if tracked_people[track_id]["missed_frames"] > 30:
                tracks_to_remove.append(track_id)

    # Remove old tracks
    for track_id in tracks_to_remove:
        start_time = datetime.strptime(tracked_people[track_id]["start_time"], "%Y-%m-%d %H:%M:%S.%f")
        end_time = datetime.now()
        time_duration = round((end_time - start_time).total_seconds(), 2)

        with open(tracking_durations_csv, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Track ID:", track_id, "Time Duration:", time_duration])

        # Write Violation to Firebase Realtime Database
        violation_data = {
            "person_id": track_id,
            "violation_duration": time_duration,
            "timestamp": datetime.now().isoformat(),
            "image_url": tracked_people[track_id].get("image_url", "")
        }

        send_data_thread = threading.Thread(target=send_violation_info_to_database, args=(violation_data,))
        send_data_thread.start()

        print(f"Violation Ended --> Person ID: {track_id} / Duration: {time_duration} sec")

        #with tracked_people_lock:
        del tracked_people[track_id]
      

    cv2.putText(frame, f"Violations: {len(tracked_people)}", 
                (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (255, 0, 0), 
                2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    display_width = 1920
    if display_width > 0:
        display_height = int((display_width / frame_width) * frame_height)
    display_frame = cv2.resize(frame, (display_width, display_height))
    cv2.namedWindow("Tracking", cv2.WINDOW_KEEPRATIO)

    cv2.imshow("Tracking", display_frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
