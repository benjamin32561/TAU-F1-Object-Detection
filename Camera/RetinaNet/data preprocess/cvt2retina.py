import dataset_functions as df
import common.functions as cf
from loguru import logger
from dataset_consts import IMAGES, ANNO, COCO_ENDING, IMAGES_DST_PATH
from common.consts import IMAGES_SUB_FOLDER, LABELS_SUB_FOLDER, \
                        SRC_PATH, DATASET_SPLIT_RATIO, OBJECTS

def main():
    phases_json_data = df.GetPhaseJsonDict()

    #id's for COCO format
    img_id = 0
    bbx_id = 0

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

            #get current phase in split
            phase = cf.GetCurrentPhase(nof, img_idx)
            dst_images = IMAGES_DST_PATH+phase+COCO_ENDING

            #save image in new location with correct format
            filename = cf.CopyImage(src_images, dst_images, original_filename)

            #load image data json file
            json_file_path = df.os.path.join(src_labels, original_filename+".json")
            img_json_data = cf.GetDataFromJson(json_file_path)

            #add image to coco dataset
            phases_json_data[phase][IMAGES].append(
                    df.GetImageDataToAdd(img_json_data, filename, img_id)
                )

            #extracting data from json file
            for bounding_box in img_json_data[OBJECTS]:
                #add bounding box to json data
                phases_json_data[phase][ANNO].append(
                        df.GetAnnoDataToAdd(img_id,bbx_id,bounding_box)
                    )
                bbx_id+=1
            img_id+=1

    #build final dict where key is full .json 
    #file name and value is final json data
    final_dict = {}
    for key,val in phases_json_data.items():
        final_dict[df.GetFullDstJson(key)] = val
    cf.WriteJsonFiles(final_dict)

if __name__ == '__main__':
    main()