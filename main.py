from fastapi import FastAPI
from typing import Dict
import cv2
import numpy as np
import Person
from ultralytics import YOLO

app = FastAPI()

# Load YOLO model
model = YOLO(r"D:\Fast api\yolov8n.pt")

@app.get('/count_people')
async def count_people() -> Dict[str, int]:
    entry_count, exit_count = process_video()
    return {'entry_count': entry_count, 'exit_count': exit_count}

def process_video():
    entry_count = 0
    exit_count = 0

    # Video input
    cap = cv2.VideoCapture(r"D:\Fast api\input video\example_01.mp4")

    # Define the label mapping
    label_mapping = {
        0: 'Person',
        1: 'Bicycle',
        2: 'Car',
        3: 'Motorbike',
        # Add mappings for other classes if needed
    }

    # Capture properties
    w = cap.get(3)
    h = cap.get(4)
    frameArea = h * w
    areaTH = frameArea / 300

    # Variables
    persons = []
    max_p_age = 1
    pid = 1

    # Lines coordinate for counting
    line_up = int(1.5 * (h / 6))
    line_down = int(3.5 * (h / 6))
    up_limit = int(0.5 * (h / 6))
    down_limit = int(4.5 * (h / 6))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:  # If frame could not be read (end of video)
            break
        frame = frame[:, 20:]

        # Object detection with YOLO
        results = model.predict(frame, conf=0.2)

        # Process YOLO detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, id in zip(boxes, ids):
                label_name = label_mapping.get(id, 'Unknown')
                if label_name != 'Person':
                    continue
                x1, y1, x2, y2 = box

                # Check if the person crosses the counting lines
                cy = int((y1 + y2) / 2)
                if cy in range(up_limit, down_limit):
                    for i in persons:
                        if abs((x1 + x2) // 2 - i.getX()) <= (x2 - x1) and abs(cy - i.getY()) <= (y2 - y1):
                            i.updateCoords((x1 + x2) // 2, cy)
                            if i.going_UP(line_down, line_up):
                                entry_count += 1
                            elif i.going_DOWN(line_down, line_up):
                                exit_count += 1
                            break
                    else:
                        p = Person.MyPerson(pid, (x1 + x2) // 2, cy, max_p_age)
                        persons.append(p)
                        pid += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    return entry_count, exit_count

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
