import cv2
import json
import csv 
import os
import argparse
import numpy as np 
import pandas as pd
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--in_semantic_json_dir', type=str, default="./datasets/rico/semantic_ann_json/",
                    help="Path to the annotation json files, containing annotations")

parser.add_argument('--in_img_dir', type=str, default="./datasets/rico/screenshots/",
                    help="Path to the directory containing screenshots")
parser.add_argument('--in_prgress_dir', type=str, default="./datasets/rico/progress/",
                    help="Path to the directory containing chacked ui- sem_img pairs")
parser.add_argument('--out_ann_cats_csv', type=str, default="./datasets/rico/ann_cat.csv",
                    help="Path to the annotation categories csv file")
parser.add_argument('--out_dir', type=str, default="./datasets/rico/",
                    help="Path to the output directory of instance maps, resized imgs, and semantic imgs.")
opt = parser.parse_args()

print("input semantic annotation jsons file at {}".format(opt.in_semantic_json_dir))
print("input imgs at {}".format(opt.in_img_dir))
print("input progress csv-s at {}".format(opt.in_prgress_dir))
print("output annotation map at{}".format(opt.out_ann_cats_csv))
print("output dir at {}".format(opt.out_dir))
print("")

'''
1. Read data from progress_x.csv, and process only those images, maintain a counter for total nb 
2. Create a map for, component gray-scale ( based on UI_details full csv)
3. Create semantic images from semantic Json - coomponent type - gray scale 
4. Create instance maps from semantic Json 
5. Rescale sem img to 360 x 640 
6. Rescale instance img to 360 x 640 
7. Downscale imput images, ususally by a factor of 4, to obtain 360 x 640 
'''

imgs_list = []
'''
1.Go through progress.csv files 
'''
for i in range(1,5):
    name = opt.in_prgress_dir + "progress_"+str(i)+".csv"
    if not Path(name).exists(): 
        print(name) 
        continue
    file_read = pd.read_csv(name)
    # print(file_read.head())
    imgs_list.extend(file_read.loc[file_read['action'] == 'keep'].ix[:,0].tolist())
    # print(keep_val[:5])
'''
'''

nb_images = len(imgs_list)
nb_removed_imgs = 0 
print (nb_images)
desired_height = 640
desired_width = 360
height = 2560
width = 1440 
def_color = 0
changed_map = False 

'''
2. Iterate through semantic jsons, create map of element types 
'''
label_color_dict = dict()
if Path(opt.out_ann_cats_csv).exists(): 
    df = pd.read_csv(opt.out_ann_cats_csv)
    if not df.empty: 
        label_color_dict = dict(zip(list(df.element),list(df.color)))

print(label_color_dict.items())

def semantic_map_generator(dict_var):
    #print( type(dict_var))
    global label_color_dict 
    global img
    global def_color
    global changed_map
    color =  def_color
    if "componentLabel" in dict_var: 
        if dict_var["componentLabel"] not in label_color_dict:
            label_color_dict[dict_var["componentLabel"]] = len(label_color_dict)
            changed_map = True
        color = label_color_dict[dict_var["componentLabel"]]
        if "bounds" in dict_var: 
            v = dict_var["bounds"]
            # print(v)
            img[v[1]:v[3],v[0]:v[2]] = color
    if "children" in dict_var: 
        # print( type(v))
        for child in dict_var["children"]: 
            #print( type(child))
            semantic_map_generator(child)

'''
3. Save semantic images. Create/verify output directory 
'''
save_imgs = True
out_folder = opt.out_dir+"semantic_ann_img_checked_val"
if not Path(out_folder).is_dir(): 
    try:  
        os.mkdir(out_folder)
    except OSError:  
        print ("Creation of the directory %s failed" % out_folder)
        save_imgs = False 
    else:  
        print ("Successfully created the directory %s " % out_folder)

resize_faults = list()
for i in range(nb_images):
    imgs_list[i] = imgs_list[i].split('.')[0]
    file_name = Path(opt.in_semantic_json_dir + str(imgs_list[i])+".json")
    if not file_name.exists(): 
        nb_removed_imgs = nb_removed_imgs + 1
        continue
    with open(file_name,"r") as read_json:
        data = json.load(read_json)
        # print (data.keys())
        img = np.zeros((height, width,1), np.uint8)
        semantic_map_generator(data)
        # save the semantic img
        if not save_imgs: 
            continue
        img2 = cv2.resize(img,(desired_width,desired_height))
        m = np.amax(img2)
        if m > 25 :
            print(imgs_list[i],m)
            resize_faults.extend(imgs_list[i])
        else: 
            cv2.imwrite(out_folder+"/"+str(imgs_list[i])+".png",img2)
            img3 = cv2.cvtColor( cv2.imread(out_folder+"/"+str(imgs_list[i])+".png"),cv2.COLOR_BGR2GRAY)
            m = cv2.countNonZero(cv2.subtract(img2,img3))
            if (img3.shape != img2.shape ): 
                print(imgs_list[i],img3.shape ," - ", img2.shape)
            elif ( m > 0):
                print(imgs_list[i],np.amax(img2), "-", np.amax(img3), "nb of diffs:", m)
                resize_faults.extend(imgs_list[i])
        
print ("modif imgs:", len(resize_faults))
# print(imgs_list[:10])
print(label_color_dict.items())
print("removed imgs:",nb_removed_imgs, "+ resize faults:", len(resize_faults))

if changed_map:
    with open(opt.out_ann_cats_csv, 'w') as csv_file: 
        csv_file.write("element,color\n")
        for key in label_color_dict.keys():
            csv_file.write("%s,%s\n"%(key,label_color_dict[key])) 
'''
4. Create instance maps from semantic Json 
'''
save_imgs = True
out_folder = opt.out_dir+"instance_map_img_checked_val"
if not Path(out_folder).is_dir(): 
    try:  
        os.mkdir(out_folder)
    except OSError:  
        print ("Creation of the directory %s failed" % out_folder)
        save_imgs = False 
    else:  
        print ("Successfully created the directory %s " % out_folder)

def instance_map_generator(dict_var):
    #print( type(dict_var))
    global img
    global color
    if "bounds" in dict_var: 
        v = dict_var["bounds"]
        # print(v)
        img[v[1]:v[3],v[0]:v[2]] = (color)
    if "children" in dict_var: 
        # print( type(v))
        for child in dict_var["children"]: 
            #print( type(child))
            color = color + 1
            instance_map_generator(child)


for i in range(nb_images):
    file_name = Path(opt.in_semantic_json_dir + str(imgs_list[i])+".json")
    if not file_name.exists(): 
        continue
    if imgs_list[i] in resize_faults:
        continue
    with open(file_name,"r") as read_json:
        data = json.load(read_json)
        # print (data.keys())
        img = np.zeros((height, width,1), np.uint8)
        color = 0
        instance_map_generator(data)
        # save the semantic img
        if not save_imgs: 
            continue
        cv2.imwrite(out_folder+"/"+str(imgs_list[i])+".png", cv2.resize(img,(desired_width,desired_height)))
        
'''
7. Downscale imput images, ususally by a factor of 4, to obtain 360 x 640 
'''
save_imgs = True
out_folder = opt.out_dir+"screenshots_resized_checked_val"
if not Path(out_folder).is_dir(): 
    try:  
        os.mkdir(out_folder)
    except OSError:  
        print ("Creation of the directory %s failed" % out_folder)
        save_imgs = False 
    else:  
        print ("Successfully created the directory %s" % out_folder)

for i in range(nb_images):
    file_name = Path(opt.in_semantic_json_dir + str(imgs_list[i])+".json")
    if not file_name.exists(): 
        continue
    if imgs_list[i] in resize_faults:
        continue
    file_name = opt.in_img_dir + str(imgs_list[i])+".jpg"
    if not Path(file_name).exists(): 
        continue
    img = cv2.imread(file_name)
    cv2.imwrite(out_folder+"/"+str(imgs_list[i])+".png", cv2.resize(img,(desired_width,desired_height)))

#verify 
count = 0
for i in range(nb_images): 
    json_file = Path(opt.in_semantic_json_dir + str(imgs_list[i])+".json")
    sem_img = Path(opt.out_dir+"semantic_ann_img/"+str(imgs_list[i])+".png")
    res_img = Path(opt.out_dir+"screenshots_resized/"+str(imgs_list[i])+".png")
    correct = False if ( not json_file.exists() and sem_img.exists()) else True
    correct = False if ( json_file.exists() and  not sem_img.exists()) else correct
    if correct is True: 
        count = count + 1

print("nb imgs in progress files",nb_images,"-",count," nb of processed images")