import dataset_functions as df
import common.functions as cf
from dataset_consts import DST_PATH
from loguru import logger
from dataset_consts import DST_IMAGES_SUB_FOLDER, DST_LABELS_SUB_FOLDER
from common.consts import IMAGES_SUB_FOLDER, LABELS_SUB_FOLDER, \
                        SRC_PATH, DATASET_SPLIT_RATIO, OBJECTS, BOUNDING_BOX_AREA_THRESH

def main():
    #creating dataset folders
    folders_to_create = []
    for phase in DATASET_SPLIT_RATIO.keys():
        folders_to_create.append(DST_PATH+DST_IMAGES_SUB_FOLDER+'/'+phase)
        folders_to_create.append(DST_PATH+DST_LABELS_SUB_FOLDER+'/'+phase)
    cf.CreateDirectories(folders_to_create)

    #getting sub folder data
    folders, n_folders = cf.GetSubFolders(SRC_PATH)
    n_folder = 1

    for src_sub_path in folders: #iterating through sub folders
        logger.info("{0}/{1} folders".format(n_folder,n_folders))
        n_folder+=1

        #image and lable sorce folders
        src_images = src_sub_path+IMAGES_SUB_FOLDER
        src_labels = src_sub_path+LABELS_SUB_FOLDER

        #loading image list and shuffeling it for random split
        files, nof = cf.GetFilesInDir(src_images)

        ten_per = nof//10
        for img_idx, original_filename in enumerate(files): #iterating through images in sub folder
            #print progress
            if img_idx%ten_per==0:
                logger.info("{0}/{1}".format(img_idx,nof))
            
            #load image data json file
            json_file_path = df.os.path.join(src_labels, original_filename+".json")
            img_json_data = cf.GetDataFromJson(json_file_path)
            if len(img_json_data[OBJECTS])==0: # image does not contain bounding boxes, skip it 
                continue

            #get current phase in split
            phase = cf.GetCurrentPhase(nof, img_idx)
            dst_images = df.os.path.join(DST_PATH+DST_IMAGES_SUB_FOLDER,phase)
            dst_labels = df.os.path.join(DST_PATH+DST_LABELS_SUB_FOLDER,phase)

            #save image in new location with correct format
            filename = cf.CopyImage(src_images, dst_images, original_filename)

            #extract image data from json
            img_h, img_w = cf.GetImgHWFromJson(img_json_data)
            img_size = img_h*img_w

            #extracting bounding box data from json file
            #creating a list of lines to write to the text file
            all_lines = []
            for bbx in img_json_data[OBJECTS]:
                size = cf.GetBbxArea(bbx)
                area_ratio = size/img_size
                if area_ratio>=BOUNDING_BOX_AREA_THRESH:
                    line = df.CreateBoundingBoxLineByYOLOFormat(bbx, img_h, img_w)
                    all_lines.append(line)
            #save labels to text file
            text_f = df.os.path.join(dst_labels, filename+".txt")
            with open(text_f,"w+") as text_file:
                text_file.write('\n'.join(all_lines))

if __name__ == '__main__':
    main()