import cv2
import numpy as np
import os
from ultralytics import YOLO

VIDEO_PATH = "C:/Users/Yu Yue/Desktop/MIE1517_project/testing_final.mp4"
RESULT_PATH = "result2.mp4"
MODEL_PATH = 'C:/Users/Yu Yue/Desktop/MIE1517_project/runs/detect/yolov8_hardhat_detection14/weights/best.pt'

if not os.path.exists(VIDEO_PATH):
    print(f"Video file does not exist at path: {VIDEO_PATH}")
    exit()

polygon = np.array([[900, 600], [1400, 340], [1400, 860], [1000, 860]])

model = YOLO(MODEL_PATH)

def is_point_inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0

def get_class_colors(class_names):
    import random
    random.seed(42)
    class_colors = {}
    for class_id in class_names:
        color = tuple([int(x) for x in np.random.choice(range(256), size=3)])
        class_colors[class_id] = color
    return class_colors

if __name__ == '__main__':
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video")
        exit()

    fps = capture.get(cv2.CAP_PROP_FPS)
    f_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (f_width, f_height))

    frame_count = 0
    total_count = 0
    unique_ids_in_polygon = set()

    class_names = model.names
    class_colors = get_class_colors(class_names)

    while True:
        success, frame = capture.read()
        if not success:
            print("Frame access failed or end of video reached.")
            break

        frame_count += 1
        results = model.track(frame, persist=True)
        annotated_frame = frame.copy()

        cv2.polylines(annotated_frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=3)
        count_inside = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names.get(cls, "Unknown")

                if class_name.lower() == "person":
                    track_id = int(box.id[0].cpu().numpy())
                else:
                    track_id = None

                color = class_colors.get(cls, (0, 255, 0))
                thickness = 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

                if class_name.lower() == "person" and track_id is not None:
                    label_text = f"{class_name} ID:{track_id}"
                else:
                    label_text = f"{class_name}"

                font_scale = 0.5
                label_thickness = 1
                (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)
                x_label = x1
                y_label = y1 - 10
                if y_label - label_height < 0:
                    y_label = y1 + label_height + 10

                cv2.rectangle(annotated_frame,
                              (x_label, y_label - label_height),
                              (x_label + label_width, y_label + baseline),
                              color,
                              cv2.FILLED)

                text_color = (255, 255, 255)
                cv2.putText(annotated_frame, label_text, (x_label, y_label),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, label_thickness)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                if is_point_inside_polygon((center_x, center_y), polygon):
                    if class_name.lower() == "person":
                        count_inside += 1
                        unique_ids_in_polygon.add(track_id)

                        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)

                        alert_text = f'Alert! ID:{track_id} in zone'
                        alert_font_scale = 0.5
                        alert_thickness = 1
                        (alert_label_width, alert_label_height), alert_baseline = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, alert_font_scale, alert_thickness)
                        alert_x_label = x_label
                        alert_y_label = y_label - 10

                        if alert_y_label - alert_label_height < 0:
                            alert_y_label = y_label + alert_label_height + 20

                        cv2.rectangle(annotated_frame,
                                      (alert_x_label, alert_y_label - alert_label_height),
                                      (alert_x_label + alert_label_width, alert_y_label + alert_baseline),
                                      (0, 0, 255),
                                      cv2.FILLED)

                        cv2.putText(annotated_frame, alert_text, (alert_x_label, alert_y_label),
                                    cv2.FONT_HERSHEY_SIMPLEX, alert_font_scale, (255, 255, 255), alert_thickness)

        cv2.putText(annotated_frame, f'# PPL Inside Area: {count_inside}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("YOLO Tracking with Polygon Count", annotated_frame)
        video_writer.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Total unique 'person' objects entered the polygon: {len(unique_ids_in_polygon)}")