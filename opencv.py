import cv2
import numpy as np
import os

VIDEO_PATH = "C:/Users/Yu Yue/Desktop/MIE1517_project/VIDEO/test.mp4"
RESULT_PATH = "result1.mp4"

import os
print("Current Working Directory:", os.getcwd())

if not os.path.exists(VIDEO_PATH):
    print(f"Video file does not exist at path: {VIDEO_PATH}")
    exit()

# polygon points
polygon= np.array([[710,200],[1110,220],[810,400],[410,400]])
if __name__ == '__main__': 
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened(): 
        print("Error opening video")
        exit()
    
    # print(capture.get(5))# getting the frame rate
    # print(capture.get(4))# getting the frame rate
    # print(capture.get(3))# getting the frame rate
    fps = capture.get(5)
    f_width = capture.get(3)
    f_hight = capture.get(4)
    video_writer = None
    while True: 
        success, frame = capture.read()
        if not success: 
            print("frame access failed")
            break
        # cv2.line(frame, (0, int(f_hight/2)), (int(f_width), int(f_hight/2)), (0,0,255),3)
        cv2.polylines(frame, [polygon], True, (0,0,255),3)
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [polygon], (0,0,255))
        frame= cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

        if video_writer is None: 
            fourcc = cv2.VideoWriter_fourcc("m","p","4","v")
            video_writer = cv2.VideoWriter(RESULT_PATH, fourcc,fps, (int(f_width), int(f_hight)))
        
        video_writer.write(frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

    capture.release() # release the resourse by capture
    video_writer.release()
    cv2.destroyAllWindows() # close all window
