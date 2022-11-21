#destination folder
DST_PATH = "/content/retinanet_dataset/"
DST_ANNO_PATH = DST_PATH+"annotations/" #annotations/ folder is COCO convention
IMAGES_DST_PATH = DST_PATH+"images/"
COCO_ENDING = "2017"
COCO_JSON_START = "instances_"

#each final .json file hase common build, can be found in this file
BASE_JSON_PATH = "base_json.json" #path to basic json path

#destination json consts, COCO format
IMAGES = "images"
ANNO = "annotations"