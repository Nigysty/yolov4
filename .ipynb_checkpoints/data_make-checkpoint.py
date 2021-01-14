import os
import cv2
from tqdm import tqdm

def crowd_convert():
    root = r'/home/yolov4-pytorch1/object-detection-crowdai'
    #out = r'D:\ubuntu_share\yolov4-pytorch\output_img'
    labels = os.path.join(root, 'labels.csv')
    fr = open(labels, 'r', newline='')
    labels_dict = {}
    num1, num2, num3 = 0, 0, 0
    newlines = []
    for i,line in enumerate(fr):
        if i == 0:
            continue
        info = line.strip().split(',')
        name = root + '/' + info[4]
        # x1 = str(float(info[0])*608/1920)
        # y1 = str(float(info[1])*608/1200)
        # x2 = str(float(info[2])*608/1920)
        # y2 = str(float(info[3])*608/1200)

        x1 = info[0]
        y1 = info[1]
        x2 = info[2]
        y2 = info[3]
        cls = 0
        if info[5] == 'Car':
            cls = 0
        elif info[5] == 'Truck':
            cls = 0
        elif info[5] == 'Pedestrian':
            cls = 1
        if name in labels_dict:
            labels_dict[name].append(','.join([x1,y1,x2,y2,str(cls)]))
        else:
            labels_dict[name] = [name, ','.join([x1,y1,x2,y2,str(cls)])]
    for lab in labels_dict:
        newline = ' '.join(labels_dict[lab])+'\n'
        newlines.append(newline)

    fw = open('/home/yolov4-pytorch1/work_dir/my_train3.txt', 'w')
    fw.writelines(newlines)

# def resize_image():
#     root = r'D:\ubuntu_share\yolov4-pytorch\object-detection-crowdai'
#     out = r'output_img'
#     os.makedirs(out, exist_ok=True)
#     img_lis = os.listdir(root)
#     for im in tqdm(img_lis):
#         im_path = os.path.join(root, im)
#         x = cv2.imread(im_path)
#         x = cv2.resize(x, (608,608))
#         cv2.imwrite(os.path.join(out, im), x)
#
#
# def highway_convert():
#     output = 'train_new.txt'
#     root = r'/data/yolo4_dataset_20200715'
#     img_dir = os.path.join(root,'images')
#     images = os.listdir(img_dir)
#     lines = []
#     for img in tqdm(images):
#         new_line = []
#         name = os.path.join(img_dir, img)
#         txt_name = name.replace('images', 'labels').replace('.jpg', '.txt')
#         fr = open(txt_name, 'r').readlines()
#         new_line.append(name)
#         img = cv2.imread(name)
#         print(img.shape)
#         h,w,_ = img.shape
#         h = float(h)
#         w = float(w)
#         for lin in fr:
#             data = lin.strip().split(' ')
#             if int(data[0]) != 0:
#                 continue
#
#             cx,cy,cw,ch = float(data[1])*w,float(data[2])*h,float(data[3])*w,float(data[4])*h
#
#             x1 = str((cx-cw/2))
#             y1 = str((cy-ch/2))
#             x2 = str((cx+cw/2))
#             y2 = str((cy+ch/2))
#             print(x1,y1,x2,y2)
#             info = [x1,y1,x2,y2, data[0]]
#             info = ','.join(info)
#             # print(info)
#             new_line.append(info)
#         #print(new_lines)
#         new_line = ' '.join(new_line) + '\n'
#         lines.append(new_line)
#         print(new_line)
#     with open('train_new.txt', 'w') as fw:
#         fw.writelines(lines)


if __name__ == '__main__':
    crowd_convert()

