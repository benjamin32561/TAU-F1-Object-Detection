import sys
import os
import numpy as np
import seaborn as sns
sys.path.append(os.path.abspath('../'))
import matplotlib.pyplot as plt
import common.functions as cf
from loguru import logger
import argparse
from common.consts import IMAGES_SUB_FOLDER, LABELS_SUB_FOLDER, \
                        SRC_PATH, OBJECTS, CLASS_TITLE, ID , \
                        TRAIN_IMAGE_SIZE, MIN_BBX_SIZE, NOISE_CLASS
                        
from common.functions import GetBbxWH, WriteJsonFiles, GetImgHWFromJson

def IsGoodBbx(box_size, img_size):
    """
    Determines whether a bounding box should be filtered based on a minimum size threshold.
    input:
        box_size (tuple): The size of the bounding box in pixels (width, height).
        img_size (tuple): The dimensions of the image in pixels (width, height).
        min_size (tuple, optional): The minimum bounding box size threshold in pixels.
    output:
        bool: True if the bounding box is good, False if it should be filtered.

    """
    # Calculate the minimum bounding box size threshold relative to the image size
    min_size_x = int(MIN_BBX_SIZE[0] * img_size[0] / TRAIN_IMAGE_SIZE[0])
    min_size_y = int(MIN_BBX_SIZE[1] * img_size[1] / TRAIN_IMAGE_SIZE[1])

    # Check if the bounding box size is greater than or equal to the minimum size threshold
    if box_size[0] >= min_size_x and box_size[1] >= min_size_y:
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser(description='My script')

    # add a --save_at argument to the parser
    parser.add_argument('--save_at', type=str, default='/content/', help='folder path to save graphs at')

    # parse the command-line arguments
    args = parser.parse_args()

    #getting sub folder data
    folders, _ = cf.GetSubFolders(SRC_PATH)
    n_folder = 1

    final_cnt = {}
    removed_cnt = {}
    file_path_id = {}
    for src_sub_path in folders: #iterating through sub folders
        n_folder+=1

        #image and lable sorce folders
        src_images = src_sub_path+IMAGES_SUB_FOLDER
        src_labels = src_sub_path+LABELS_SUB_FOLDER

        #loading image list and shuffeling it for random split
        files, _ = cf.GetFilesInDir(src_images)

        for original_filename in files: #iterating through images in sub folder

            #load image data json file
            json_file_path = os.path.join(src_labels, original_filename+".json")
            img_json_data = cf.GetDataFromJson(json_file_path)

            for bbx in img_json_data[OBJECTS]:
                class_title = bbx[CLASS_TITLE]
                if class_title not in final_cnt.keys():
                    final_cnt[class_title] = 1
                    removed_cnt[class_title] = 0
                else:
                    final_cnt[class_title]+=1
                
                img_h, img_w = GetImgHWFromJson(img_json_data)
                is_good = IsGoodBbx(GetBbxWH(bbx),(img_w,img_h))
                if is_good or class_title in NOISE_CLASS:
                    if json_file_path not in file_path_id.keys():
                        file_path_id[json_file_path] = []
                    file_path_id[json_file_path].append(bbx[ID])
                    removed_cnt[class_title]+=1

    print("Class Distrebution before clean: ")

    print(final_cnt)# create a barplot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    ax = sns.barplot(x=list(final_cnt.keys()), y=list(final_cnt.values()))
    # set the labels and title
    ax.set(xlabel='Class Names', ylabel='Number of Objects', title='Class Object Count')
    # display the graph
    plt.savefig(os.path.join(args.save_at, "Class Dist Before BBX Clean.png"))

    # for file_path in file_path_id.keys(): #deleting bbx from files
    #     img_json_data = cf.GetDataFromJson(file_path)
    #     old_img_bbx = img_json_data[OBJECTS]
    #     bbx_id_to_rem = file_path_id[file_path]
    #     img_json_data[OBJECTS] = [d for d in old_img_bbx if d[ID] not in bbx_id_to_rem]
    #     WriteJsonFiles({file_path:img_json_data})
    
    print("\nClass Distrebution after clean: ")

    final_cnt = {key: final_cnt[key] - removed_cnt[key] for key in final_cnt}

    print(final_cnt)
    # create a barplot using Seaborn
    plt.clf()
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    ax = sns.barplot(x=list(final_cnt.keys()), y=list(final_cnt.values()))
    # set the labels and title
    ax.set(xlabel='Class Names', ylabel='Number of Objects', title='Class Object Count')
    # display the graph
    plt.savefig(os.path.join(args.save_at, "Class Dist After BBX Clean.png"))

if __name__ == '__main__':
    main()