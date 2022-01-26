from locale import strcoll
from tensorflow.keras.models import load_model
# import keras_ocr
from easyocr import Reader
# from utils import ocr
from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import preprocessing, nn_matching
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from core.config import cfg
from tensorflow.python.saved_model import tag_constants
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging
# import pytesseract
import tensorflow as tf
import time
import os

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# deep sort imports

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def ocr(roi,reader):
    resimg = cv2.resize(roi, None, fx=25, fy=25,
                        interpolation=cv2.INTER_CUBIC)
    imgBlur = cv2.GaussianBlur(resimg, (3, 3), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(imgGray)

    try:
        number = results[0][-2]
        prob = results[0][-1]
        if number == 0:
            pass
        elif tid in d:
            if d[tid][1] < prob:
                # d[tid].append([name, prob])
                d[tid] = [number, prob]
        else:
            d[tid] = [number, prob]
    except:
        pass
    return number


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    d = {}
    # pipeline = keras_ocr.pipeline.Pipeline()
    reader = Reader(['en'], gpu=True)
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(
            FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num += 1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(
                output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # update tracks

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            if class_name == 'person':
                ocr = True
                name = None

                # draw bbox on screen
                # # Local
                # boundaries = [([155, 155, 155], [255, 255, 255]),  # white
                #               ([0, 0, 0], [100, 100, 100]),  # Black
                #               ([70, 205, 110], [140, 255, 170]),  # Green - Ref
                #               ]

                # BAYERN VS DORT
                # boundaries = [([0, 0, 115], [40, 40, 160]),  # red
                #               ([0, 200, 190], [50, 255, 255]),  # yellow
                #               ([0, 0, 0], [40, 40, 40])]  # Black

                # RMA VS CHEL
                boundaries = [([70, 45, 15], [145, 100, 100]),  # blue
                              ([210, 220, 220], [255, 255, 255]),  # white
                              ([150, 150, 160], [200, 255, 255])]  # Ref - yellow

                sumList = []
                roi = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

                try:
                    # -- Team Color Detection
                    for (lower, upper) in boundaries:
                        lower = np.array(lower, dtype="uint8")
                        upper = np.array(upper, dtype="uint8")

                        mask = cv2.inRange(roi, lower, upper)
                        output = cv2.bitwise_and(roi, roi, mask=mask)

                        sum = output.any(axis=-1).sum()
                        sumList.append(sum)

                    i = np.argmax(sumList)

                    TeamList = ['whi', 'bla', 'Ref']
                    team = TeamList[i]

                    # color_list = [(30, 30, 150), (40, 225, 225), (15, 15, 15)]
                    color_list = [
                        (230, 230, 230), (20, 20, 20), (100, 230, 140)]
                    color = color_list[i]

                    tid = str(track.track_id)

                    if team == 'Ref':
                        cv2.putText(img, 'Ref', (int(bbox[0]), int(bbox[1] - 7)), 0, 0.5,
                                    color, 2)
                        continue

                    if tid in d:
                        if d[tid][1] > 0.9:
                            number = ocr(roi, reader, tid)
                            ocr = False
                    else:
                        number = ocr(roi, reader, tid)

                    rma = ['1','3','4','6','7','8','9','10','13','14','15','18','20','22','23','24','33']
                    che = ['2','4','5','6','7','11','12','13','15','16','19','21','22','24','25','26','28','29','30','31','35']

                    # OCR
                    if ocr == True:
                        number = None
                        resimg = cv2.resize(roi, None, fx=25, fy=25,
                                            interpolation=cv2.INTER_CUBIC)
                        imgBlur = cv2.GaussianBlur(resimg, (3, 3), 1)
                        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
                        results = reader.readtext(imgGray)

                        try:
                            number = results[0][-2]
                            prob = results[0][-1]
                            if number.isnumeric():
                                if number == 0 or number == '0':
                                    pass
                                elif tid in d:
                                    if d[tid][1] < prob:
                                        # d[tid].append([name, prob])
                                        d[tid] = [number, prob]
                                else:
                                    d[tid] = [number, prob]
                        except:
                            pass

                    if tid in d:
                        name = str(d[tid][0]) + "-" + str(team)
                    else:
                        name = str(team)

                    # track.track_id = str(number) + "-" + str(number)
                    cv2.putText(img, name, (int(bbox[0]), int(bbox[1] - 7)), 0, 0.5,
                                color, 2)
                    if FLAGS.info:
                        print("Tracker ID: {}, Class: {}, name: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, name, (
                            int(
                                bbox[0]),
                            int(
                                bbox[1]),
                            int(
                                bbox[2]),
                            int(bbox[3]))))
                except:
                    pass

        # calculate frames per second of running detections
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(img)
        # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # if not FLAGS.dont_show:
        #     cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
