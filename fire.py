import cv2
import time
from detection import utils
import numpy as np
# from Fire_tracking.fire_detect.realsense import *
# from Fire_tracking.fire_detect.loading_circle_bar import *
focal = 875.81
# f = Loading_CircleBar
def angle_calculation(x_c, y_c, x, y):
    hor_angle = np.arcsin(np.abs(x_c - x) / focal) * 180 / np.pi
    ver_angle = np.arcsin(np.abs(y_c - y) / focal) * 180 / np.pi
    dia_angle = np.arcsin(np.sqrt((x_c - x) ** 2 + (y_c - y) ** 2) / focal) * 180 / np.pi
    return hor_angle, ver_angle, dia_angle


def execute(frame, model):
    # Convert to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Predict
    img_np = np.array(image)
    results = utils.detect_fn(img_np, model)
    num_detections = int(results.pop('num_detections'))
    results = {key: value[0, :num_detections].numpy()
               for key, value in results.items()}
    results['num_detections'] = num_detections
    # Check need mask
    need_mask, results = utils.check_need_mask(results, 0.5)

    cmap = [(25, 25, 182), (0, 255, 0), (11, 94, 235)]

    fh, fw, fc = image.shape
    x = 0
    y = 0
    for label, bbox, score in results:
        y1, x1, y2, x2 = bbox
        y1 = int(y1 * fh)
        x1 = int(x1 * fw)
        y2 = int(y2 * fh)
        x2 = int(x2 * fw)
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        score = round(score, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), cmap[label], 4)
        cv2.putText(frame, "{}, {}".format(label, score), (x - 60, y - 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    point = (x, y)
    cv2.circle(frame, point, radius= 3, color = (0, 0, 255), thickness= 2)
    return frame, point


path_dict = {
    'checkpoint': 'configs/my_ckpt/ckpt-2',
    'pipeline': 'configs/pipeline.config',
    'label_map': 'configs/Fire-Smoke_label_map.pbtxt'
}
model = utils.load_model(path_dict)
# dc = DepthCamera()
cam = cv2.VideoCapture("E:/Robot-Vision-master/fire8.avi")
xc = 640
yc = 360
while True:
    start_time = time.time()
    # ret, depth, color = dc.get_frame()
    ret, color = cam.read()
    # cv2.imshow("real", color)
    color, points = execute(color, model)
    cv2.circle(color, (xc, yc), radius= 3, color = (255, 0, 0), thickness=3)
    # rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # distance = depth[points[1], points[0]]
    # cv2.putText(color, "{}mm".format(distance), (points[0], points[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0),2)
    ha, va, da = angle_calculation(xc,yc , points[0], points[1])
    cv2.putText(color, "Angle: {}".format(ha), (80, 400), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    cv2.imshow("frame", color)
    # print("====", 1 / (time.time() - start_time))
    key = cv2.waitKey(3) & 0xFF
    if key & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()