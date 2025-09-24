from ultralytics import YOLO
import cv2
from sort import Sort
import numpy as np
import logging
import time
from datetime import datetime
import os
import matlab.engine
import csv
import threading

from pytapo import Tapo
# Replace with your camera's details
host = "192.168.137.98"  # Camera IP
username = "duldulemre"
password = "duldulemre"
cam = Tapo(host, "admin", "E.a1n2t3")
alarm_on = False


# Event used to signal the alarm thread
alarm_stop_event = threading.Event()

def alarm_worker():
    # ensure loud enough
    cam.setSpeakerVolume(8)
    # Loop until we’re told to stop
    while not alarm_stop_event.is_set():
        # (re)start the user audio clip at index 0
        cam.testUsrDefAudio(1, True)
        # brief pause so we don't slam the camera API —
        # you can set this to your file's length (in seconds)
        time.sleep(1)
    # once stop_event is set, kill playback
    cam.testUsrDefAudio(2, True)
# —————————————————————————————————————————————


"""
# ----------------------- Firebase Initialization -----------------------
import firebase_admin
from firebase_admin import credentials, db, storage


# Replace this JSON file name with the path to your actual service account key file.
firebase_cred = credentials.Certificate("033d5490-7329-4414-bc9d-539019e63cb1.json")


# IMPORTANT: Use your actual Realtime Database URL and storageBucket here.
firebase_admin.initialize_app(firebase_cred, {
    'databaseURL': 'https://duldul-2c57f-default-rtdb.firebaseio.com/',
    'storageBucket': 'duldul-2c57f.firebasestorage.app'  # Replace with your Firebase bucket
})
"""

tracked_people_lock = threading.Lock()
"""
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
"""


logging.getLogger("ultralytics").setLevel(logging.CRITICAL)


# Initialize CSV file
total_positives_csv = "total_positive_detections.csv"
with open(total_positives_csv, mode='w', newline='') as file:
    csv_writer = csv.writer(file)


# Initialize CSV file
tracking_durations_csv = "tracking_durations.csv"
with open(tracking_durations_csv, mode='w', newline='') as file:
    csv_writer = csv.writer(file)



# Video source
stream = 'rtsp://duldulemre:duldulemre@192.168.137.98/stream1'
# Video file or webcam source
VIDEO_FILE = "Videos/TestVideo1.mp4"


# Initialize video capture
cap = cv2.VideoCapture(stream)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
resize_width = 1920
resize_height = int((resize_width / frame_width) * frame_height) if frame_width > 0 else 480
print("Actual Frame Width x Height: ", frame_width,"x",frame_height)
print("Resized Frame Width x Height: ", resize_width,"x",resize_height)



success, frame = cap.read()

if success:
    #frame = cv2.resize(frame, (1920, 1080))
    frame = cv2.resize(frame, (resize_width, resize_height))
    # Convert BGR (OpenCV default) frame to RGB for MATLAB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Initialize MATLAB engine
    eng = matlab.engine.start_matlab()
    # Convert OpenCV frame to MATLAB array
    matlab_frame = matlab.uint8(frame_rgb.tolist())  # Convert to MATLAB format
    # Call MATLAB function to extract walkway area (Polygon)
    walkway_area = eng.detect_walkway_from_frame(matlab_frame, nargout=1)
    # Convert walkway_area back to NumPy array for use in Python code

    # Ensure walkway_area is in NumPy array format with the correct shape
    walkway_area = np.array(walkway_area, dtype=np.int32)  # Ensure it is in integer format
    walkway_area = walkway_area[:, [1, 0]]
    walkway_area = walkway_area.astype(np.int32)
    eng.quit()


    cv2.polylines(frame, [walkway_area], True, (0, 255, 255), 2)
    while True:
        # Enable manual resizing of the OpenCV window
        cv2.namedWindow("Select Area", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Select Area", frame)
        key = cv2.waitKey(0)
        if key == 13:  # Wait for Enter to proceed
            cap.release()
            break
else:
    print("Error reading the video.")
    exit()



# Initialize YOLO and SORT
chosen_model = YOLO("yolo11l.pt").to("cuda")  # YOLO model

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)  # SORT tracker
# max_age: Number of frames to wait before a missing object is removed
# min_hits: Minimum consecutive frames an object must appear to be tracked
# iou_threshold: Minimum overlap required to associate a detection with an existing track

# Directory to save videos of tracked people
output_video_dir = "Violations"
if not os.path.exists(output_video_dir):
    os.makedirs(output_video_dir)


# Restart the video from the beginning
cap = cv2.VideoCapture(stream)

tracked_people = {}


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
    results = chosen_model.predict(frame, classes=[0], conf=0.4, device="cuda")
    violation_detections = []

    total_people = []

    # Draw the walkway area
    cv2.polylines(frame, [walkway_area], True, (0, 255, 255), 2)


    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()  # Bounding box for the detected person
            conf = box.conf[0].cpu().numpy()  # Confidence score for the detected person
            
            # Check if the bottom-center of the bounding box is within the walkway area
            in_walkway = cv2.pointPolygonTest(walkway_area, (int((int(x_min) + int(x_max)) / 2), int(y_max)), False)
            total_people.append(box)
            # in_walkway >= 0 if the person is inside the walkway
            if in_walkway >= 0:
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            # in_walkway < 0 if the person is outside the walkway
            else:
                violation_detections.append([x_min, y_min, x_max, y_max, conf])
    
    #Write only the number of violations to the CSV
    with open(total_positives_csv, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([len(total_people)])  # Log violations count

    # Convert to NumPy array (required by SORT)
    violation_detections = np.array(violation_detections)

    if len(violation_detections) > 0:
        tracked_objects = tracker.update(violation_detections)
    else:
        tracked_objects = []

    # IDs of people currently outside the walkway in the current frame
    people_outside_walkway = [] 

    # Draw tracking results on the frame
    for obj in tracked_objects:
        x_min, y_min, x_max, y_max, track_id = obj
        track_id = int(track_id)
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        cv2.putText(frame, f"ID: {int(track_id)}", (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frame, (int((int(x_min)+int(x_max))/2), int(y_max)), 4, (255,0,0),-1)
        

        people_outside_walkway.append(track_id)


        # Expand the bounding box by 50 pixels in all directions
        x_min = max(0, int(x_min) - 50)
        y_min = max(0, int(y_min) - 50)
        x_max = min(resize_width, int(x_max) + 50)
        y_max = min(resize_height, int(y_max) + 50)


        # Crop the frame to enclose the tracked person in video recording
        cropped_frame = frame[y_min:y_max, x_min:x_max]
        cropped_frame = cv2.resize(cropped_frame, (track_frame_width, track_frame_height))
        

        if track_id not in tracked_people:
            tracked_people[track_id] = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                "missed_frames": 0,  # Initialize missed frames counter
            }
            
            # Save a single image when the violation is first detected
            output_filename = f"{output_video_dir}/trackID_{track_id}_violation.jpg"
            cv2.imwrite(output_filename, cropped_frame)
            """
            # Start a new thread for uploading the image
            upload_thread = threading.Thread(target=upload_to_firebase, args=(output_filename, track_id, tracked_people))
            upload_thread.start()
            """
            print(f"Walkway Violation Detected --> Person ID: {track_id} exited walkway at {tracked_people[track_id]['start_time']}")

        else:
            # Reset missed frames counter if detected again
            tracked_people[track_id]["missed_frames"] = 0


# In your main loop, replace your old start/stop calls with:

    if len(tracked_people) > 0 and not alarm_on:
        # clear any previous stop request
        alarm_stop_event.clear()
        # launch the thread (daemon so it won’t block program exit)
        threading.Thread(target=alarm_worker, daemon=True).start()
        alarm_on = True
        #print("🚨 Alarm turned ON")

    elif len(tracked_people) == 0 and alarm_on:
        # signal the thread to stop the siren
        alarm_stop_event.set()
        alarm_on = False
        #print("✅ Alarm turned OFF")


    # List to hold tracks that should be removed
    tracks_to_remove = [] 

    for track_id in list(tracked_people.keys()):

        if track_id not in people_outside_walkway:

            # Increment missed frames counter for people not detected in the current frame
            tracked_people[track_id]["missed_frames"] += 1

            # If missed frames exceed the limit, mark for removal
            if tracked_people[track_id]["missed_frames"] > 30:
                tracks_to_remove.append(track_id)


    # Remove tracks that exceeded max missed frames limit
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
        """
        send_data_thread = threading.Thread(target=send_violation_info_to_database, args=(violation_data,))
        send_data_thread.start()
        """

        print(f"Violation Ended --> Person ID: {track_id} / Violation Duration: {time_duration} sec")

        with tracked_people_lock:
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
    # Enable manual resizing of the OpenCV window
    cv2.namedWindow("Tracking", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Tracking", display_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
