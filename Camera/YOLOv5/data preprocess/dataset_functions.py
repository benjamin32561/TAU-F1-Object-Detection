import sys
import os
sys.path.append(os.path.abspath('../../'))
import common.consts as cc
import common.functions as cf
from model_consts import CONE_CLASS_LABEL

"""
description: creates a line representing a bounding box in
            the specified YOLOv5 format (view dataset_format.txt for more information)
input:
    bbx - bounding box json data from original json file
    h - image height
    w - image width
output:
    str - the final line to add to the .txt file
"""
def CreateBoundingBoxLineByYOLOFormat(bbx:dict, h:int, w:int):
    class_num = str(CONE_CLASS_LABEL[bbx[cc.CLASS_TITLE]])
    points = bbx[cc.POINTS][cc.EXT]
    x1,y1 = points[0][0],points[0][1]
    x2,y2 = points[1][0],points[1][1]
    bb_width = x2-x1
    bb_height = y2-y1
    x_center = bb_width/2 + x1
    y_center = bb_height/2 + y1

    return ' '.join([class_num,
                    str(x_center/w),
                    str(y_center/h),
                    str(bb_width/w),
                    str(bb_height/h)])