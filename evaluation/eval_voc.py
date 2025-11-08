import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, threshold=1.0):
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
            

            cal = gt < 255
            mask = (predict == gt) * cal

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
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T, threshold))
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
        print('%11s:%7.1f%%' % ('mIoU', miou * 100))
    return loglist




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', type=str)
    parser.add_argument("--predict_dir", default='results/voc/masks/', type=str)
    parser.add_argument("--gt_dir", default='dataset/VOCdevkit/VOC2012/SegmentationClass', type=str)

    args = parser.parse_args()

    with open(args.list,'r') as f:
        name_list = [line.strip() for line in f if line.strip()]


    loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)