import cv2
import torch
from shapely.geometry import Point, Polygon
import numpy as np
import time

# Load YOLOv5 model (pretrained on COCO)
model = torch.hub.load('yolov5', 'yolov5s', source='local')
model.classes = [0]  # Only detect humans (class 0 in COCO)

# Load video (or use 0 for webcam)
cap = cv2.VideoCapture(0)

# Read first frame to get dimensions
ret, frame = cap.read()
if not ret:
    print("Failed to read from camera.")
    cap.release()
    exit()

height, width = frame.shape[:2]

# Define polygon (danger zone) as left half of the frame
danger_zone_points = [(0, 0), (width // 2, 0), (width // 2, height), (0, height)]
danger_zone = Polygon(danger_zone_points)

# Sidebar settings
sidebar_width = 250
log_entries = []
entry_counter = 0
person_in_zone = False
zone_entry_time = None
zone_duration = 0
scroll_offset = 0
entries_per_page = 10  # Number of entries to show at once

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Draw danger zone
    cv2.polylines(frame, [np.array(danger_zone.exterior.coords, np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # Track if any person is in the zone this frame
    current_in_zone = False

    # Extract detections
    detections = results.xyxy[0].cpu().numpy() if results.xyxy[0].numel() > 0 else []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        point = Point(center_x, center_y)

        # Check if inside polygon
        if danger_zone.contains(point):
            color = (0, 0, 255)
            label = "ALERT"
            current_in_zone = True
        else:
            color = (0, 255, 0)
            label = "Safe"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (center_x, center_y), 4, color, -1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Entry and timing logic
    if current_in_zone and not person_in_zone:
        entry_counter += 1
        zone_entry_time = time.time()
        entry_time_str = time.strftime("%H:%M:%S", time.localtime(zone_entry_time))
        log_entries.append({"entry": f"Entry #{entry_counter}", "duration": 0, "time": entry_time_str})
    elif not current_in_zone and person_in_zone and zone_entry_time is not None:
        # Person just left the zone, update duration
        zone_duration = time.time() - zone_entry_time
        log_entries[-1]["duration"] = zone_duration
        zone_entry_time = None

    # If person is still in zone, update duration live
    if current_in_zone and zone_entry_time is not None:
        log_entries[-1]["duration"] = time.time() - zone_entry_time

    person_in_zone = current_in_zone

    # Handle scrolling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 82:  # Up arrow
        if scroll_offset > 0:
            scroll_offset -= 1
    elif key == 84:  # Down arrow
        if scroll_offset < max(0, len(log_entries) - entries_per_page):
            scroll_offset += 1

    # Add sidebar to frame
    sidebar = np.ones((height, sidebar_width, 3), dtype=np.uint8) * 220
    cv2.rectangle(sidebar, (0, 0), (sidebar_width, height), (200, 200, 200), -1)
    cv2.putText(sidebar, "Danger Zone Log", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Show only a window of entries
    visible_entries = log_entries[scroll_offset:scroll_offset + entries_per_page]
    for idx, entry in enumerate(visible_entries):
        y_base = 60 + idx * 70
        cv2.putText(sidebar, entry["entry"], (10, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(sidebar, f"At: {entry['time']}", (10, y_base + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(sidebar, f"Time: {entry['duration']:.1f}s", (10, y_base + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Scrollbar indicator
    if len(log_entries) > entries_per_page:
        bar_height = int(height * entries_per_page / len(log_entries))
        bar_pos = int(height * scroll_offset / len(log_entries))
        cv2.rectangle(sidebar, (sidebar_width - 15, bar_pos), (sidebar_width - 5, bar_pos + bar_height), (100, 100, 100), -1)

    # Concatenate sidebar to frame
    frame_with_sidebar = np.concatenate((frame, sidebar), axis=1)

    # Show frame
    cv2.imshow('YOLO + Danger Zone', frame_with_sidebar)

cap.release()
cv2.destroyAllWindows()
