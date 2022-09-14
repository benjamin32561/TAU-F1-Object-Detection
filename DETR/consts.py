import sys, os
sys.path.append(os.path.abspath('../'))
from common import *

#different for each model
#cone class-label dict
CONE_CLASS_LABEL = {
    "blue_cone":1,
    "yellow_cone":2,
    "orange_cone":3,
    "large_orange_cone":4,
    "other_cone":5
}

#destination folder
DST_PATH = "../detr_dataset/"
DST_ANNO_PATH = DST_PATH+"annotations/" #annotations/ folder is COCO convention

#each final .json file hase common build, can be found in this file
BASE_JSON_PATH = "base_json.json" #path to basic json path

#destination json consts, COCO format
IMAGES = "images"
ANNO = "annotations"



def GetImageDataToAdd(img_json_data:dict,short_filename:str,img_id:int):
    #extract image data from json
    img_h, img_w = GetImgWHFromJson(img_json_data)
    #all string are COCO consts and appear only here
    return {
            "license": 1,
            "file_name": short_filename+IMG_EXTEN,
            "coco_url": "",
            "height": img_h,
            "width": img_w,
            "date_captured": "",
            "flickr_url": "",
            "id": img_id
        }

def GetAnnoDataToAdd(img_id:int,bbx_id:int,bbx:dict):
    class_num = CONE_CLASS_LABEL[bbx[CLASS_TITLE]]
    points = bbx[POINTS][EXT]
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

def GetPhaseJsonDict():
    phases_json_data = {}
    #load base data for json files
    base_json_object = GetDataFromJson(BASE_JSON_PATH)
    #create directories
    dirs_to_create = [DST_ANNO_PATH]
    for phase in PHASES.keys(): #creating destination folders to save images
        dirs_to_create.append(DST_PATH+phase)
        phases_json_data[phase]=copy.deepcopy(base_json_object) #setting base json data of each file
    CreateDirectories(dirs_to_create)
    return phases_json_data

def GetFullDstJson(phase:str):
    return DST_ANNO_PATH+phase+".json"