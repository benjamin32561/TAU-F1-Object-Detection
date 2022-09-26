import sys
import os
sys.path.append(os.path.abspath('../'))
import common_consts as cc
import common_functions as cf
import dataset_consts as dc
from model_consts import CONE_CLASS_LABEL

"""
description: extracts image data from given json data to add to COCO dataset
            (view dataset_format.txt for more information)
input:
    img_json_data - image json data (from original file)
    short_filename - image file name without extention
    img_id - image id (COCO format)
output: 
    dict - the final image data to add to the COCO format dataset
"""
def GetImageDataToAdd(img_json_data:dict,short_filename:str,img_id:int):
    #extract image data from json
    img_h, img_w = cf.GetImgHWFromJson(img_json_data)
    #all string are COCO consts and appear only here
    return {
            "license": 1,
            "file_name": short_filename+cc.IMG_EXTEN,
            "coco_url": "", #irrelevent
            "height": img_h,
            "width": img_w,
            "date_captured": "", #irrelevent
            "flickr_url": "", #irrelevent
            "id": img_id
        }

"""
description: extracts image annotation (bounding boxs) data from given json data 
            to add to COCO dataset (view dataset_format.txt for more information)
input:
    img_id - image id (COCO format)
    bbx_id - bounding box id (COCO format)
    bbx - bounding box data from the original json file
output:
    dict - the final bounding box data to add to the COCO format dataset
"""
def GetAnnoDataToAdd(img_id:int,bbx_id:int,bbx:dict):
    class_num = CONE_CLASS_LABEL[bbx[cc.CLASS_TITLE]]
    points = bbx[cc.POINTS][cc.EXT]
    x1,y1 = points[0][0],points[0][1]
    x2,y2 = points[1][0],points[1][1]
    bb_width = x2-x1
    bb_height = y2-y1
    #all string are COCO consts and appear only here
    return {
            "image_id": img_id,
            "area":bb_width*bb_height,
            "bbox": [
                x1, #x1
                y1, #y1
                bb_width, #width
                bb_height #height
            ],
            "category_id": class_num,
            "id": bbx_id
        }

"""
description: creates the dataset directories and a dict
            (view dataset_format.txt for more information about the directories)
input:
output:
    dict - dict that contains phases as keys and the
    basic json format for the COCO dataset
"""
def GetPhaseJsonDict():
    phases_json_data = {}
    #load base data for json files
    base_json_object = cf.GetDataFromJson(dc.BASE_JSON_PATH)
    #create directories
    dirs_to_create = [dc.DST_ANNO_PATH]
    for phase in cc.DATASET_SPLIT_RATIO.keys(): #creating destination folders to save images
        dirs_to_create.append(dc.DST_PATH+phase)
        phases_json_data[phase]=cf.copy.deepcopy(base_json_object) #setting base json data of each file
    cf.CreateDirectories(dirs_to_create)
    return phases_json_data

"""
description: builds the final and full phase json file path
input:
    phase - the phase
output:
    string - full json file path
"""
def GetFullDstJson(phase:str):
    return dc.DST_ANNO_PATH+phase+".json"