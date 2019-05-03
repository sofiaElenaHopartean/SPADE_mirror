import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str, default="./")
opt= parser.parse_args()

print("scanning directory {}".format(opt.in_dir))
print("")

types_dict = dict()
folder_name = opt.in_dir

for root, dirs, files in os.walk(folder_name):
    # print('Found directory: %s' % root)
    for name in files: 
        # print('\t%s' % name)
        split = name.split('.')
        key = split[0]
        if (len(split) > 1): 
            key = '.'.join(split[1:])
        key2 = root.split(folder_name)[1]+"/"+key
        if key2 in types_dict: 
            types_dict[key2] = types_dict[key2] +1
        else: 
            types_dict[key2] = 1 
        
for key, value in types_dict.items():
    print(key," :",value)