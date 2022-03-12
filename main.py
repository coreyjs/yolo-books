import cv2

from yolo_network import YoloNetwork, CaptureDevice


def main():
    cap = cv2.VideoCapture(0)
    yolo = YoloNetwork(img_size=320, convidence_threshold=0.5, nms_threshold=0.3,
                        c_device=CaptureDevice.ZERO)
    # instantiate our NN architecture, load weights + config
    yolo.load_classes()
    yolo.load_model()

    capture = cv2.VideoCapture(0)

    # capture loop
    skip = True
    while True:
        # limit our processing to the skip rate, we dont need to process this date every loop
        # iteration (todo this could be way better)
        skip = not skip
        if skip: continue
        
        success, img = capture.read()

        # Run forward pass through our network
        outputs = yolo.forward(image_data=img)

        # Run object detection using the output from the NN (.forward())
        # This runs NMS (non maximum supression) to limit our bounding boxes
        yolo.find_objects(outputs=outputs, img=img)
        yolo.show()

        cv2.waitKey(1)

if __name__ == "__main__":
    main()