import cv2
import numpy as np
import time
from progress.bar import Bar
from utils import templateMatching, featureMatching
import argparse

methods = [
    'cv2.TM_CCOEFF',
    'cv2.TM_CCOEFF_NORMED',
    'cv2.TM_CCORR_NORMED',
    'cv2.TM_SQDIFF',
    'cv2.TM_SQDIFF_NORMED'
    ]

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str, required=True,
    help="path to input videos")
ap.add_argument("-t", "--template", type=str, required=True,
    help="path to input template")
ap.add_argument("-m", "--method", type=str, default='cv2.TM_SQDIFF_NORMED', choices=methods,
    help="type method matching")
args = vars(ap.parse_args())

vc = cv2.VideoCapture(args['input'])
template = cv2.imread(args['template'])
h, w, _ = template.shape
method = eval(args['method'])
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(template,None)
dataTemplate = (template, kp1, des1)
totalFrame = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
runningTime = 0.0
with Bar('Processing',max=totalFrame) as bar:
    while True:
        bar.next()
        ret, img = vc.read()
        if not ret:
            break
        start = time.time()
        top_left = templateMatching(template,img,method)
        bottom_right = (top_left[0] + w, top_left[1] + h)
        new = img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
        kp2, des2 = sift.detectAndCompute(new,None)
        dataImage = (new, kp2, des2)
        dst, good, matches = featureMatching(dataTemplate,dataImage,5)
        runningTime += time.time()-start
        if type(dst) is np.ndarray:
            cv2.rectangle(img,top_left, bottom_right, 255, 2)
        else:
            cv2.putText(img,'MISSING', (int(w/2),int(h/2)), cv2.FONT_HERSHEY_SIMPLEX ,1, (0,0,255), 4, cv2.LINE_AA) 
        cv2.imshow(args['method'],img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
runningTime = time.time()-start
print(f'INFO fps: {totalFrame/runningTime}')