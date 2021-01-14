import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# Head

# output: (B,A*n_ch,H,W)--->(B,A,H,W,n_ch)
# output为模型输出的
def yolo_decode(output, num_classes, anchors, num_anchors, scale_x_y):
    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()
        # 检查device

    n_ch = 4+1+num_classes
    # n_ch:[tx,tx,tw,th,obj,nclasses]
    A = num_anchors
    # A = 3
    B = output.size(0)
    H = output.size(2)
    W = output.size(3)

    output = output.view(B, A, n_ch, H, W).permute(0,1,3,4,2).contiguous()
    # 下面索引的都是n_ch
    bx, by = output[..., 0], output[..., 1]
    # bx, by = tx, ty
    bw, bh = output[..., 2], output[..., 3]
    # bw, bh = tw, th

    det_confs = output[..., 4]
    # det_confg:置信度
    cls_confs = output[..., 5:]
    # cls_confg:检测物体类别的one-hot码

    bx = torch.sigmoid(bx)
    by = torch.sigmoid(by)
    bw = torch.exp(bw)*scale_x_y - 0.5*(scale_x_y-1)
    bh = torch.exp(bh)*scale_x_y - 0.5*(scale_x_y-1)
    # 针对不同尺度大小的同一类物体，用scale_x_y来解决
    # 默认scale_x_y=1，因此默认bw = torch.exp(bw)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)
    # 概率化
 
    # [0,1,2,3,...,18] (若网格大小为19*19)
    grid_x = torch.arange(W, dtype=torch.float).repeat(1, 3, W, 1).to(device)
    # 将W[0，1，...，18]作为1个维度，向前（1，3，W）三个维度复制，得到（1，3，19，19）
    grid_y = torch.arange(H, dtype=torch.float).repeat(1, 3, H, 1).permute(0, 1, 3, 2).to(device)
    # grid_y与grid_w相比，只是比grid_x多个一个最后一维的行列互换
    # grid_y与grid_w：其实都是对物体当前所在网格的一个绝对网格序号索引
    bx += grid_x
    by += grid_y
    # bx,by  （以物体所在的网格为单位的中心点bx，by） 即相对距离

    for i in range(num_anchors):
        bw[:, i, :, :] *= anchors[i*2]
        bh[:, i, :, :] *= anchors[i*2+1]

    bx = (bx / W).unsqueeze(-1) #  物体的中心点（bx，by）相对于整张图的偏移量 （即绝对距离）
    # unsqueeze(-1) ： （1，3，19，19）---> （1，3，19，19，1）
    by = (by / H).unsqueeze(-1)
    bw = (bw / W).unsqueeze(-1)
    bh = (bh / H).unsqueeze(-1)

    #boxes = torch.cat((x1,y1,x2,y2), dim=-1).reshape(B, A*H*W, 4).view(B, A*H*W, 1, 4)
    boxes = torch.cat((bx, by, bw, bh), dim=-1).reshape(B, A * H * W, 4)
    # (1,3,19,19,4)--->(1,3*19*19,4)
    det_confs = det_confs.unsqueeze(-1).reshape(B, A*H*W, 1)
    cls_confs =cls_confs.reshape(B, A*H*W, num_classes)
    # confs = (det_confs.unsqueeze(-1)*cls_confs).reshape(B, A*H*W, num_classes)
    outputs = torch.cat([boxes, det_confs, cls_confs], dim=-1)


    #return boxes, confs
    return outputs


class YoloLayer(nn.Module):
    ''' Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    '''
    def __init__(self, img_size, anchor_masks=[], num_classes=80, anchors=[], num_anchors=9, scale_x_y=1):
        super(YoloLayer, self).__init__()
        #[6,7,8]
        self.anchor_masks = anchor_masks
        #类别 默认为80类（COCO数据集
        self.num_classes = num_classes
        # [12,16,19,36,40,28,36,75,76,55,72,146,142,110,192,243,459,401]
        if type(anchors) == np.ndarray:
            self.anchors = anchors.tolist()
        else:
            self.anchors = anchors

        print(self.anchors)
        print(type(self.anchors))
        
        # 9 总的anachors数（1个head对应3个anchors）
        self.num_anchors = num_anchors
        # 18/9 =2 ： 表示在anchors中每两个表示一个anchor的w，h
        self.anchor_step = len(self.anchors) // num_anchors
        print(self.anchor_step)
        self.scale_x_y = scale_x_y
        # scale_x_y=1 缩放因子（默认为1）

        self.feature_length = [img_size[0]//8,img_size[0]//16,img_size[0]//32]
        self.img_size = img_size

    def forward(self, output):
        if self.training:
            return output

        in_w = output.size(3)
        anchor_index = self.anchor_masks[self.feature_length.index(in_w)]
        stride_w = self.img_size[0] / in_w
        # 32 stride：对应yolohead的网格大小，在这anchor_mask=[6,7,8],对应的yolohead的网格大小为32*32
        masked_anchors = []
        for m in anchor_index:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
            # 索引由[6，7，8]对应的anchors，得到[142，110，192，243，459，401]
        # [142,110,192,243,459,401]/32
        # 作用：上面得到的[142，110，192，243，459，401]为像素值
        # 将它除以stride，可得到网格单位，它等于[aWi/stride , aHi/stride]
        self.masked_anchors = [anchor / stride_w for anchor in masked_anchors]
        # output:(B,A*n_ch,H,W)--->(1,3*(5+80),19,19)

        data = yolo_decode(output, self.num_classes, self.masked_anchors, len(anchor_index),scale_x_y=self.scale_x_y)
        return data



