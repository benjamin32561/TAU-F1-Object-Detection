import json
import os
import shutil
import cv2

"""
input dataset directories should be like that:
dataset/.../images/ - contains the images
dataset/.../labels/ - contains the original .json files from the downloaded dataset

output dataset hirarchy will be:
yolov5_dataset/images/ - contains .jpg images
yolov5_dataset/labels/ - contains .txt files
the relationship between the files is as specified in the formats.txt file
"""

cone_class_label = {
    "blue_cone":0,
    "yellow_cone":1,
    "orange_cone":2,
    "large_orange_cone":3,
    "other_cone":4
}

src_path = "dataset/amz/" #dataset/.../
src_images = src_path+"images/"
src_labels = src_path+"labels/"
dst_images = "yolov5_dataset/images"
dst_labels = "yolov5_dataset/labels"
os.makedirs(dst_images, exist_ok=True)
os.makedirs(dst_labels, exist_ok=True)

nof = len(os.listdir(src_images))
ten_per = nof//10
i = 0
for original_filename in os.listdir(src_images):
    #print progress
    if i%ten_per==0:
        print("{0}/{1}".format(i,nof))
    i+=1

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

    #extracting data from json file
    #creatign a list of lines ti write to the text file
    all_lines = []
    for bounding_box in json_data["objects"]:
        class_num = str(cone_class_label[bounding_box["classTitle"]])
        points = bounding_box["points"]["exterior"]
        x1,y1 = points[0][0],points[0][1]
        x2,y2 = points[1][0],points[1][1]
        bb_width = x2-x1
        bb_height = y2-y1
        x_center = bb_width/2 + x1
        y_center = bb_height/2 + y1

        line = ' '.join([class_num, str(x_center/img_width), str(y_center/img_height), str(bb_width/img_width), str(bb_height/img_height)])
        all_lines.append(line)

    #save labels to text file
    text_f = os.path.join(dst_labels, filename+".txt")
    with open(text_f,"w+") as text_file:
        text_file.write('\n'.join(all_lines))

