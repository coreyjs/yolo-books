from enum import Enum
import numpy as np
import cv2

class CaptureDevice(Enum):
    ZERO = 0
    ONE = 1


class YoloNetwork:
    def __init__(self, img_size: int = 320, convidence_threshold: float = 0.5,
                 nms_threshold: float = 0.3, c_device: CaptureDevice = CaptureDevice.ZERO,
                 net_config: str = 'yolov3-320.cfg', net_weights: str = 'yolov3.weights', verbose: bool = False) -> None:
        self.img_size = img_size
        self.convidence_threshold = convidence_threshold
        self.nms_threshold = nms_threshold
        self.c_device = c_device
        self.net_config = net_config
        self.net_weights = net_weights
        self.verbose = verbose

        self.nn = None
        self.class_names = []
        self.output_layers = None
        self.image = None
    
    def _blob_from_image(self, img):
        return cv2.dnn.blobFromImage(img, 1/255, (self.img_size, self.img_size), [0, 0, 0], 1, crop=False)

    def load_classes(self, file_name: str = 'coco.names'):
        self.class_names = []
        with open('coco.names', 'rt') as f:
            self.class_names = f.read().rstrip('\n').split('\n')

    def load_model(self, backend: str = cv2.dnn.DNN_BACKEND_OPENCV,
                    target: str = cv2.dnn.DNN_TARGET_OPENCL) -> None:
        self.nn = cv2.dnn.readNetFromDarknet(self.net_config, self.net_weights)
        self.nn.setPreferableBackend(backend)
        self.nn.setPreferableTarget(target)
        self.output_layers = self.nn.getUnconnectedOutLayersNames()
    
    def forward(self, image_data):
        self.image = image_data
        blob = self._blob_from_image(image_data)
        self.nn.setInput(blob)
        return self.nn.forward(self.output_layers)

    def show(self):
        cv2.imshow('Image', self.image)

    def find_objects(self, outputs, img, limit_objects = None) -> None:
        if self.nn is None:
            raise Exception(message='Network not initialized, call load_model')
        
        hT, wT, cT = img.shape
        bb, class_ids, convidence = [], [], []

        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                convidence_value = scores[class_id]

                if convidence_value > self.convidence_threshold:
                    print(class_id)
                    # if we are limiting our detection algo to certain objects
                    # then we can skip ones we do not care about
                    if limit_objects is not None and class_id not in limit_objects:
                        continue
                    w, h = int(detect[2] * wT), int(detect[3] * hT) # pixel values
                    x, y = int((detect[0] * wT) - w/2), int((detect[1] * hT) - h/2)
                    bb.append([x, y, w, h])
                    class_ids.append(class_id)
                    convidence.append(float(convidence_value))
       #print(f'Bounding Boxes: {len(bb)}')

        # run non max supression on our bounding boxes
        indices = cv2.dnn.NMSBoxes(bb, convidence, self.convidence_threshold, self.nms_threshold)

        for i in indices:
            bounding_box = bb[i]
            x, y, w, h = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(
                img, 
                f'{self.class_names[class_ids[i]].upper()} {int(convidence[i] * 100)}%',
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 0, 255),
                2 #thickness
            )