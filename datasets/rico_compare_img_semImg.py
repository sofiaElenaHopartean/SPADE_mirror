import cv2
import csv 
import os
import argparse
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--in_img_dir', type=str, default="./rico/screenshots/",
                    help="Path to the directory containing screenshots")
parser.add_argument('--in_semantic_img_dir', type=str, default="/media/sofiahopartean/EC2A2B102A2AD6FC/rico-images/semantic_imgs/",
                    help="Path to the directory containing screenshots")
parser.add_argument('--in_prgress_dir', type=str, default="./rico/progress/",
                    help="Path to the directory containing chacked ui- sem_img pairs")
opt = parser.parse_args()

print("input images at {}".format(opt.in_img_dir))
print("input semantic annotation images at {}".format(opt.in_semantic_img_dir))
print("input progress csv-s at {}".format(opt.in_prgress_dir))
print("")

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

imgs_list.sort()

'''
2. Create new csv file 
'''
out_file_path = opt.in_prgress_dir+"progress_5.csv"
listing = os.listdir(opt.in_semantic_img_dir)
missed_count = 0 

with open(out_file_path, mode='w') as progress_csv: 
    progress_csv.write("image,action\n")

    #iterate over images and semImgs 
    for img_id in listing:
        sem_img = mpimg.imread(opt.in_semantic_img_dir+str(img_id))
        img_path = opt.in_img_dir+str(img_id)
        if not Path(img_path).exists(): 
            missed_count = missed_count +1  
            continue
        img = mpimg.imread(img_path)

        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(sem_img)
        f.add_subplot(1,2, 2)
        plt.imshow(np.rot90(img,2))
        plt.show(block=True)

        key = input("keep?(y/n)")
        action = "keep" if key == 'y' else "drop"
        progress_csv.write("%s,%s\n"%(img_id,action))
