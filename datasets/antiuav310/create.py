#将现有的label格式由【cls，x_center，y_center，w，h】转变为【cls，x1，y1，w1，h1，x2，y2，w2，h2，x_center，y_center，w，h】
import os

label_path = '/data/users/qinhaolin01/TIP/OSFormer/datasets/antiuav310/labels_ori'

total = 5
sample = 3
save_name = 'labels_' + str(total) + '_' + str(sample)
image_t = [0] * sample
image_t_txt = ['data'] * sample
dx = (total - 1)//(sample-1)

for options in ['train/', 'test/', 'val/']:
    path = os.path.join(label_path, options)
    for s in os.listdir(path):
        if s.endswith("cache"):
            continue
        path_ = path + s
        path_new = path_.replace('labels_ori', save_name)
        if not os.path.exists(path_new):
            os.makedirs(path_new)
        for ss in os.listdir(path_):
            txt_path = path_ + '/' + ss
            txt_new_path = path_new + '/' + ss

            id = ss.split('.txt')[0]

            image_t[-1] = int(id)
            image_t_txt[-1] = txt_path
            for idx in range(sample-2, -1, -1):
                image_t[idx] = max((image_t[-1] - dx*(sample-1-idx)), 1)
                image_t[idx] = str(image_t[idx]).zfill(len(id))
                image_t_txt[idx] = path_ + '/' + image_t[idx] + '.txt'
                if not os.path.exists(image_t_txt[idx]):
                    image_t_txt[idx] = image_t_txt[idx+1]

            with open(image_t_txt[0], 'r') as f_t1:
                line_t1 = f_t1.readline()
            with open(image_t_txt[1], 'r') as f_t2:
                line_t2 = f_t2.readline()
            with open(image_t_txt[2], 'r') as f_t3:
                line_t3 = f_t3.readline()

            t1_box = line_t1.split(' ')
            t2_box = line_t2.split(' ')
            t3_box = line_t3.split(' ')
            t3_new_box = []

            for string in t1_box:
                t3_new_box.append(string)
            
            for i in range(1,len(t2_box)):
                t3_new_box.append(t2_box[i])
            
            for i in range(1,len(t3_box)):
                t3_new_box.append(t3_box[i])

            strs = ' '
            f_files = open(txt_new_path,'w')
            f_files.write(strs.join(t3_new_box))
            f_files.close()