import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse

categories = [    
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",]


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=81, threshold=1.0):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start, step, TP, P, T, threshold):
        for idx in range(start, len(name_list), step):
            name = name_list[idx]
            gt_file = os.path.join(gt_folder, '%s.png' % name)
            gt = np.array(Image.open(gt_file))

            width, height = Image.open(gt_file).size 
            
            predict_file = os.path.join(predict_folder, '%s.png' % name)
            predict = np.array(Image.open(predict_file).resize((width, height)))  # cv2.imread(predict_file)

            
            gt_file = os.path.join(gt_folder, '%s.png' % name)
            gt = np.array(Image.open(gt_file))
            cal = gt < 255
            mask = (predict == gt) * cal

            # print(predict.shape, gt.shape)

            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict == i) * cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt == i) * cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt == i) * mask)
                TP[i].release()

    p_list = [] 
    for i in range(20):
        p = multiprocessing.Process(target=compare, args=(i, 20, TP, P, T, threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    for i in range(num_cls):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        T_TP.append(T[i].value / (TP[i].value + 1e-10))
        P_TP.append(P[i].value / (TP[i].value + 1e-10))
        FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100

    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    printlog = True
    if printlog:
        for i in range(num_cls):
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))
    return loglist




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_dir", default='results/coco/masks', type=str)
    parser.add_argument("--gt_dir", default='dataset/coco_stuff164k/annotation_object_p/val2017', type=str)
    args = parser.parse_args()

    predict_dir = args.predict_dir
    name_list = [f.replace(".jpg","") for f in os.listdir('dataset/coco_stuff164k/images/val2017')]
    loglist = do_python_eval(predict_dir, args.gt_dir, name_list, 81)
