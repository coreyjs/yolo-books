import time
import cv2

FPS_30 = 33
FPS_24 = 42

def calc_fps(frame, prev_frame_time, current_frame_time):
    # calculating the FPS
    # FPS is the number of frames processed in a given time frame
    
    delta = (current_frame_time - prev_frame_time) * 1000
    fps = 1 / (current_frame_time - prev_frame_time)
    prev_frame_time = current_frame_time
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'FPS: {int(fps)}', (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

    return prev_frame_time, delta