#将json文件转换为txt

import json
import os

train_jsons = "/data3/publicData/Anti-UAV310/train/"
test_jsons = "/data3/publicData/Anti-UAV310/test/"
val_jsons = "/data3/publicData/Anti-UAV310/val/"
jsons = [train_jsons, test_jsons, val_jsons]

labels_path = '/data/users/qinhaolin01/Small_Detection/YOLO_CT/datasets/AntiUAV310/labels_ori/'
for paths in ['train', 'test', 'val']:
    path = os.path.join(labels_path, paths)
    if not os.path.exists(path):
        os.makedirs(path)

images_path = '/data/users/qinhaolin01/Small_Detection/YOLO_CT/datasets/AntiUAV310/images/'
files_path = '/data/users/qinhaolin01/Small_Detection/YOLO_CT/datasets/AntiUAV310/'

for json_ in jsons:
    files_txt = os.path.join(files_path, json_.split('/')[-2] + '.txt')
    f_files = open(files_txt,'w')

    files = os.listdir(json_)
    for file in files:
        IRjson = json_ + file + '/IR_label.json'
        imglist = os.listdir(json_.replace('Anti-UAV310','Anti-UAV310/Anti-UAV') + file + '/')#.remove('IR_label.json')#.sort(key= lambda x:int(x[:-10]))
        imglist.remove('IR_label.json')
        imglist.sort(key= lambda x:int(x[:-4]))
        with open(IRjson, 'r') as f:
            dataset = json.load(f)
        gtlist = dataset['gt_rect']
        assert len(imglist) == len(gtlist)

        for i in range(len(imglist)):
            bbox = gtlist[i]
            file_name = imglist[i]
            classes = 0
            h = 512
            w = 640
            x_1 = max((bbox[0]+(bbox[2]/2)),0)
            y_1 = max((bbox[1]+(bbox[3]/2)),0)
            # AntiUAV中原始label存储的是目标左上角点xy，在yolo系列中label需要中心点xy，因此需要做转换
            strs = str(classes) + ' ' + str(x_1/w) + ' ' + str(y_1/h) + ' ' + str(bbox[2]/w) + ' ' + str(bbox[3]/h)
            path1 = os.path.join(labels_path, json_.split('/')[-2], file)
            path2 = file_name.replace('jpg','txt')
            label_txt = os.path.join(path1, path2)
            if not os.path.exists(path1):
                os.makedirs(path1)
            with open(label_txt, 'w') as f_txt:
                f_txt.write(strs)
            path3 = os.path.join(images_path, json_.split('/')[-2], file, file_name)
            f_files.write(path3 + '\n')
    f_files.close()