from multiprocessing.connection import wait
import cv2
import time

from yolo_network import YoloNetwork, CaptureDevice
from utilities import calc_fps, FPS_24, FPS_30

def main():
    # limit our detection to certain objects. The row id of the class name -1 for 0 offset
    object_to_locate_class_idxs = [73]

    cap = cv2.VideoCapture(0)
    yolo = YoloNetwork(img_size=320, convidence_threshold=0.5, nms_threshold=0.3,
                        c_device=CaptureDevice.ZERO)

    # instantiate our NN architecture, load weights + config
    yolo.load_classes()
    yolo.load_model()

    capture = cv2.VideoCapture(0)

    prev_frame_time = time.time()
    frames_per_second_limit = FPS_24
    while cap.isOpened():
        
        success, frame = capture.read()

        if not success:
            break

        prev_frame_time, delta = calc_fps(frame=frame, prev_frame_time=prev_frame_time, current_frame_time=time.time())
        #print(f'Delta: {delta} ms')
        #cv2.imshow('Image', frame)
        if delta < frames_per_second_limit:
            # sleep for timeoffset to try to maintain a constant 30fps
            time.sleep( (frames_per_second_limit - delta) / 1000)

        # Run forward pass through our network
        outputs = yolo.forward(image_data=frame)

        # Run object detection using the output from the NN (.forward())
        # This runs NMS (non maximum supression) to limit our bounding boxes
        yolo.find_objects(outputs=outputs, img=frame, limit_objects=object_to_locate_class_idxs)
        yolo.show()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()