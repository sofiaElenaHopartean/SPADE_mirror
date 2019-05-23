from os import listdir
import argparse
import cv2 
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, default="./datasets/rico/semantic_ann_img_checked_val/")
opt= parser.parse_args()

print("scanning directory {}".format(opt.in_dir))
print("")

folder_name = opt.in_dir
count = 0 

# 44390 
img3 = cv2.cvtColor( cv2.imread(folder_name+"44390.jpg"),cv2.COLOR_BGR2GRAY)
print(img3)

for files in listdir(folder_name):
    
    split = files.split('.')
    if(split[-1]=="jpg"): 
        m1 = np.amax(cv2.imread(folder_name+files))
        m2 = np.amax(cv2.imread(folder_name+files,cv2.IMREAD_ANYDEPTH))
        if (m1 > 25 or m2 > 25):
            print(files, m1, m2)
            count = count + 1

print ("total nb of imgs with pixel values above 25 is:",count)
