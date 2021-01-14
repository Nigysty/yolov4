# -------------------------------------#
#       创建YOLO类
# -------------------------------------#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from yolo4 import YoloBody
from utils.utils import *
from yolo_layer import *


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
# --------------------------------------------#
class Inference(object):
    # ---------------------------------------------------#
    #   初始化模型和参数，导入已经训练好的权重
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.model_path = kwargs['model_path']
        self.anchors_path = kwargs['anchors_path']
        self.classes_path = kwargs['classes_path']
        self.model_image_size = kwargs['model_image_size']
        self.confidence = kwargs['confidence']
        self.cuda = kwargs['cuda']

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()
        print(self.anchors)
        self.net = YoloBody(3, len(self.class_names)).eval()
        self.load_model_pth(self.net, self.model_path)

        if self.cuda:
            self.net = self.net.cuda()
            self.net.eval()

        print('Finished!')

        self.yolo_decodes = []
        anchor_masks = [[0,1,2],[3,4,5],[6,7,8]]
        for i in range(3):
            head = YoloLayer(self.model_image_size, anchor_masks, len(self.class_names),
                                               self.anchors, len(self.anchors)//2).eval()
            self.yolo_decodes.append(head)


        print('{} model, anchors, and classes loaded.'.format(self.model_path))

    def load_model_pth(self, model, pth):
        print('Loading weights into state dict, name: %s' % (pth))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pth, map_location=device)
        matched_dict = {}
        """
        #-------------
        for k,v in model_dict.items():
            if k.find('backbone') == -1:
                key = 'backbone'+k
            else:
                key = k
            print(key)
            print("###########")
            if np.shape(pretrained_dict[key]) == np.shape(v):
                matched_dict[k] = v
        #--------------
            
        """
        for k, v in pretrained_dict.items():
            if np.shape(model_dict[k]) == np.shape(v):
                matched_dict[k] = v
            else:
                print('un matched layers: %s' % k)
            print('...........')
            print(k)
        
        print(len(model_dict.keys()), len(pretrained_dict.keys()))
        print('%d layers matched,  %d layers miss' % (
        len(matched_dict.keys()), len(model_dict) - len(matched_dict.keys())))
        model_dict.update(matched_dict)
        model.load_state_dict(pretrained_dict)
        print('Finished!')
        return model

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return anchors
        #return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_src):
        h, w, _ = image_src.shape
        image = cv2.resize(image_src, (608, 608))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = np.array(image, dtype=np.float32)
        img = np.transpose(img / 255.0, (2, 0, 1))
        images = np.asarray([img])

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        print(output.shape)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                               conf_thres=self.confidence,
                                               nms_thres=0.6)
            
        boxes = [box.cpu().numpy() for box in batch_detections]
        print(boxes[0])
        return boxes[0]


if __name__ == '__main__':
    params = {
        #"model_path": 'pth/yolo4_weights_my.pth',
        #"model_path": 'chk/Epoch_047_Loss_11.0913.pth',
        "model_path": 'chk_dark/Epoch_050_Loss_7.7722.pth',
        "anchors_path": 'work_dir/yolo_anchors_coco.txt',
        "classes_path": 'work_dir/car_pepole.txt',
        "model_image_size": (608, 608, 3),
        "confidence": 0.4,
        "cuda": True
    }

    model = Inference(**params)
    class_names = load_class_names(params['classes_path'])
    image_src = cv2.imread('test.jpg')
    boxes = model.detect_image(image_src)
    plot_boxes_cv2(image_src, boxes, savename='output5.jpg', class_names=class_names)