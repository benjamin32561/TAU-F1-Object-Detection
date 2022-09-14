import json
import os
import shutil
from glob import glob
import random

#cone class-label dict
cone_class_label = {
    "blue_cone":1,
    "yellow_cone":2,
    "orange_cone":3,
    "large_orange_cone":4,
    "other_cone":5
}

#tvt split
phases = {"train":0.75,"val":0.15,"test":0.1}
assert sum(phases.values())==1

#paths
src_path = "../dataset/" #../dataset/.../
dst_path = "../detr_dataset/"
dst_anno_path = dst_path+"annotations/"

#json data
phases_json_data = {}

#load base data for json files
f = open('base_json.json','r')
base_json_object = json.load(f)
f.close()
#create directories
os.makedirs(dst_anno_path, exist_ok=True)
for phase in phases.keys(): #creating destination folders to save images
    os.makedirs(dst_path+phase, exist_ok=True)
    phases_json_data[phase]=base_json_object #setting base json data of each file

#id's are importent to COCO format
image_id = 0
bbx_id = 0
folders = glob(src_path+"*/", recursive = True) #getting sub folders in the original dataset folder (amz,ampera etc.)
n_folders = len(folders)
n_folder = 0
for src_sub_path in folders: #iterating through sub folders
    print("{0}/{1} folders".format(n_folder,n_folders))
    n_folder+=1
    src_images = src_sub_path+"images/"
    src_labels = src_sub_path+"labels/"

    #loading file list and shuffeling it for random split
    files = os.listdir(src_images)
    random.shuffle(files)

    nof = len(files)
    ten_per = nof//10
    image_num = 0
    for original_filename in files: #iterating through images in sub folder
        #print progress
        if image_num%ten_per==0:
            print("{0}/{1}".format(image_num,nof))

        phase = ""
        per_sum = 0
        for key in phases.keys():
            per_sum+=phases[key]
            if image_num <= per_sum*nof:
                dst_images = dst_path+key
                phase = key
                break

        #save image in new location with correct format
        f = os.path.join(src_images, original_filename)
        filename = original_filename[:-4]
        new_f = os.path.join(dst_images, filename+".jpg")
        shutil.copyfile(f, new_f)


        #load json file
        json_file_path = os.path.join(src_labels, original_filename+".json")
        json_file = open(json_file_path)
        json_data = json.load(json_file)
        json_file.close()

        #extract image data from json
        img_height = json_data["size"]["height"]
        img_width = json_data["size"]["width"]

        #add image to coco dataset
        phases_json_data[phase]["images"].append({
                "license": 1,
                "file_name": filename+".jpg",
                "coco_url": "",
                "height": img_height,
                "width": img_width,
                "date_captured": "",
                "flickr_url": "",
                "id": image_id
            })

        #extracting data from json file
        for bounding_box in json_data["objects"]:
            class_num = cone_class_label[bounding_box["classTitle"]]
            points = bounding_box["points"]["exterior"]
            x1,y1 = points[0][0],points[0][1]
            x2,y2 = points[1][0],points[1][1]
            bb_width = x2-x1
            bb_height = y2-y1

            #add bounding box to json data
            phases_json_data[phase]["annotations"].append({
                "image_id": image_id,
                "area":bb_width*bb_height,
                "bbox": [
                    x1, #x1
                    y1, #y1
                    bb_width, #width
                    bb_height #height
                ],
                "category_id": class_num,
                "id": bbx_id
            })

            bbx_id+=1
        image_num+=1
        image_id+=1

#write final json data to files
for phase in phases.keys():
    file_anno_path = dst_anno_path+phase+".json"
    with open(file_anno_path,'w+') as json_file:
        json.dump(phases_json_data[phase],json_file,indent=4)