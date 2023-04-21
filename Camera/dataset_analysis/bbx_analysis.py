import sys
import os
import cv2
sys.path.append(os.path.abspath('../'))
import common.functions as cf
from common.consts import POINTS, EXT

def ShowImage(full_img_path, bbxs, save=False, show=True):
    img = cv2.imread(full_img_path)
    for bbx in bbxs:
        start_end = bbx[POINTS][EXT]
        x1,y1 = start_end[0][0],start_end[0][1]
        x2,y2 = start_end[1][0],start_end[1][1]
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    if show:
        cv2.imshow(full_img_path,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save:
        cv2.imwrite("img.jpg", img)