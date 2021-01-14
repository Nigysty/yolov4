import os
import sys
sys.path.append(r'/home/yolov4-pytorch1/')
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import train_dataset_collate, test_dataset_collate, TrainDataset, TestDataset
from utils.generator import TrainGenerator, TestGenerator
from yolo_loss import YOLOLoss
from yolo4 import YoloBody
from yolo_layer import YoloLayer
from tqdm import tqdm

from darknet.darknet import *
from easydict import EasyDict
from config import Cfg
from Evaluation.map_eval_pil import compute_map
from tensorboardX import SummaryWriter
from utils.utils import *


Cfg.train_data = 'work_dir/my_train3.txt'
Cfg.anchors_path = 'work_dir/yolo_anchors.txt'
Cfg.classes_path = 'work_dir/my_classes.txt'
#Cfg.pth_path = 'pth/yolo4_weights_my.pth'
Cfg.pth_path = 'chk/Epoch_047_Loss_11.0913.pth'
Cfg.check = 'chk'

Cfg.use_data_loader = True
Cfg.first_train = False

Cfg.cur_epoch = 0
Cfg.total_epoch = 80
Cfg.freeze_mode = False

#valid
Cfg.valid_mode = False
Cfg.confidence = 0.4
Cfg.nms_thresh = 0.3
Cfg.draw_box = True
Cfg.save_error_miss = True
Cfg.input_dir = r'/home/yolov4-pytorch1/object-detection-crowdai'
Cfg.save_err_mis = False



#调用Evaluation模块, 进行map计算和类别准召率计算
def make_labels_and_compute_map(infos, classes, input_dir, save_err_miss=False):
    out_lines,gt_lines = [],[]
    out_path = 'Evaluation/out.txt'
    gt_path = 'Evaluation/true.txt'
    foutw = open(out_path, 'w')
    fgtw = open(gt_path, 'w')
    for info in infos:
        out, gt, shapes = info
        for i, images in enumerate(out):
            for box in images:
                bbx = [box[0]*shapes[i][1], box[1]*shapes[i][0], box[2]*shapes[i][1], box[3]*shapes[i][0]]
                bbx = str(bbx)
                cls = classes[int(box[6])]
                prob = str(box[4])
                img_name = os.path.split(shapes[i][2])[-1]
                line = '\t'.join([img_name, 'Out:', cls, prob, bbx])+'\n'
                out_lines.append(line)

        for i, images in enumerate(gt):
            for box in images:
                bbx = str(box.tolist()[0:4])
                cls = classes[int(box[4])]
                img_name = os.path.split(shapes[i][2])[-1]
                line = '\t'.join([img_name, 'Out:', cls, '1.0', bbx])+'\n'
                gt_lines.append(line)

    foutw.writelines(out_lines)
    fgtw.writelines(gt_lines)
    foutw.close()
    fgtw.close()

    args = EasyDict()
    args.annotation_file = 'Evaluation/true.txt'
    args.detection_file = 'Evaluation/out.txt'
    args.detect_subclass = False
    args.confidence = 0.2
    args.iou = 0.3
    args.record_mistake = True
    args.draw_full_img = save_err_miss
    args.draw_cut_box = False
    args.input_dir = input_dir
    args.out_dir = 'out_dir'
    Map = compute_map(args)
    return Map



# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])
    # return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def gen_lr_scheduler(lr, cur_epoch, model):
    init_lr = lr*pow(0.9, cur_epoch)
    print('init learning rate:', init_lr)
    optimizer = optim.Adam(model.parameters(), init_lr, weight_decay=5e-4)
    if Cfg.cosine_lr:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return lr_scheduler,optimizer


def gen_burnin_lr_scheduler(lr, cur_batch, model):
    # learning rate setup
    def burnin_schedule(i):
        i = i+1
        if i < Cfg.burn_in:
            factor = pow(i / Cfg.burn_in, 4)
        elif i < Cfg.steps[0]:
            factor = 1.0
        elif i < Cfg.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor

    if Cfg.TRAIN_OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'initial_lr': lr}],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif Cfg.TRAIN_OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            [{'params': model.parameters(), 'initial_lr': lr}],
            lr=lr,
            momentum=Cfg.momentum,
            weight_decay=Cfg.decay,
        )
    else:
        print('optimizer must be adam or sgd...')
        return None,None
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule, last_epoch=cur_batch-1)
    print('update learning rate:', scheduler.get_last_lr()[0])
    return scheduler, optimizer


def get_train_lines(train_data):
    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(train_data) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    return lines, num_train, num_val


def load_model_pth(model, pth, cut=None):
    print('Loading weights into state dict, name: %s'%(pth))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pth, map_location=device)

    #431: backbone
    #467: backbone+SPP
    #509: backbone+SPP+1_concat
    #557: 对应darknet的137
    # for i, key in enumerate(pretrained_dict):
    #     print('items:', i, key, np.shape(pretrained_dict[key]))
    # assert 0

    match_dict = {}
    print_dict = {}
    if cut== None: cut = (len(pretrained_dict) - 1)
    try:
        for i, (k, v) in enumerate(pretrained_dict.items()):
            if i <= cut:
                assert np.shape(model_dict[k]) == np.shape(v)
                match_dict[k] = v
                print_dict[k] = v
            else:
                print_dict[k] = '[NO USE]'

    except:
        print('different shape with:', np.shape(model_dict[k]), np.shape(v), '  name:', k)
        assert 0

    for i, key in enumerate(print_dict):
        value = print_dict[key]
        print('items:', i, key, np.shape(value) if type(value) != str else value)

    model_dict.update(match_dict)
    model.load_state_dict(model_dict)
    print('Finished!')
    return model


def get_epoch_by_name(pth_path):
    try:
        pth = pth_path
        epoch = os.path.split(pth)[-1].split('_')[1]
        epoch = int(epoch)
    except Exception as e:
        print(e, 'start epoch: %d'%Cfg.cur_epoch)
        return Cfg.cur_epoch
    return epoch


def print_model(model):
    model_dict = model.state_dict()
    for i, key in enumerate(model_dict):
        print('model items:', i, key, '---->', np.shape(model_dict[key]))


def find_pth_by_epoch(epoch):
    pth_list = os.listdir('check_point')
    for name in pth_list:
        curpo = name.split('_')[1]
        if curpo == '%03d'%(epoch):
            return os.path.join('check_point', name)
    return ''


def valid(epoch_lis, classes, draw=True, cuda=True, anchors=[]):
    writer = SummaryWriter(log_dir='valid_logs',flush_secs=60)
    epoch_size_val = num_val // gpu_batch

    model = YoloBody(len(anchors[0]), num_classes)

    anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    yolo_decodes = []
    anchors = anchors.reshape([-1])
    for i in range(3):
        head = YoloLayer((Cfg.width, Cfg.height), anchor_masks, len(classes),
                         anchors, anchors.shape[0] // 2).eval()
        yolo_decodes.append(head)

    if Use_Data_Loader:
        val_dataset = TestDataset(lines[num_train:], (input_shape[0], input_shape[1]))
        gen_val = DataLoader(val_dataset, batch_size=gpu_batch, num_workers=8, pin_memory=True,
                             drop_last=True, collate_fn=test_dataset_collate)
    else:
        gen_val = TestGenerator(gpu_batch, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate()

    for epo in epoch_lis:
        pth_path = find_pth_by_epoch(epo)
        if not pth_path:
            print('pth_path is error...')
            return False
        model = load_model_pth(model, pth_path)
        cudnn.benchmark = True
        model = model.cuda()
        model.eval()
        with tqdm(total=epoch_size_val, mininterval=0.3) as pbar:
            infos = []
            for i, batch in enumerate(gen_val):
                images_src, images, targets, shapes = batch[0], batch[1], batch[2], batch[3]
                with torch.no_grad():
                    if cuda:
                        images_val = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    else:
                        images_val = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    outputs = model(images_val)

                    output_list = []
                    for i in range(3):
                        output_list.append(yolo_decodes[i](outputs[i]))
                    output = torch.cat(output_list, 1)
                    batch_detections = non_max_suppression(output, len(classes),
                                                           conf_thres=Cfg.confidence,
                                                           nms_thres=Cfg.nms_thresh)
                    boxs = [box.cpu().numpy() for box in batch_detections]
                    # boxs = utils.post_processing(images_val, Cfg.confidence, Cfg.nms_thresh, outputs)
                    infos.append([boxs, targets, shapes])

                    if draw:
                        for x in range(len(boxs)):
                            os.makedirs('result_%d'%epo, exist_ok=True)
                            savename = os.path.join('result_%d'%epo, os.path.split(shapes[x][2])[-1])
                            plot_boxes_cv2(images_src[x], boxs[x], savename=savename, class_names=class_names)
                pbar.update(1)
            print()
            print('===========================================================================================================')
            print('++++++++cur valid epoch %d, pth_name: %s++++++++'%(epo, pth_path))
            Map = make_labels_and_compute_map(infos, classes, Cfg.input_dir, save_err_miss=Cfg.save_err_mis)
            writer.add_scalar('MAP/epoch', Map, epo)
            print()

    return True


def train(cur_epoch, Epoch, cuda=True, anchors=[]):
    #使用tensorboardX来可视化训练指标
    writer = SummaryWriter(log_dir='train_logs',flush_secs=60)

    # 创建模型
    model = YoloBody(len(anchors[0]), num_classes)

    #cut param:
    #431: backbone
    #467: backbone+SPP
    #509: backbone+SPP+1_concat
    #557: 对应darknet的137
    if Cfg.first_train:
        model = load_model_pth(model, pth_path, cut=431)
    else:
        model = load_model_pth(model, pth_path)

    cudnn.benchmark = True
    model = model.cuda()
    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes,
                                    (input_shape[1], input_shape[0]), smoooth_label))

    #lr_scheduler, optimizer = gen_lr_scheduler(lr, cur_epoch, model)
    lr_scheduler, optimizer = gen_burnin_lr_scheduler(lr, cur_batch, model)

    if Use_Data_Loader:
        train_dataset = TrainDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic)
        gen = DataLoader(train_dataset, batch_size=gpu_batch, num_workers=8, pin_memory=True,
                         drop_last=True, collate_fn=train_dataset_collate)
    else:
        gen = TrainGenerator(gpu_batch, lines[:num_train],
                        (input_shape[0], input_shape[1])).generate(mosaic=mosaic)

    epoch_size = max(1, num_train // gpu_batch)

    for epoch in range(cur_epoch, Epoch):
        total_loss = 0
        cur_step = 0
        with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            model.train()
            start_time = time.time()
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                images, targets = batch[0], batch[1]
                with torch.no_grad():
                    if cuda:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                outputs = model(images)
                losses = []
                losses_loc = []
                losses_conf = []
                losses_cls = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets)
                    losses.append(loss_item[0])
                    losses_loc.append(loss_item[3])
                    losses_conf.append(loss_item[1])
                    losses_cls.append(loss_item[2])

                loss = sum(losses) / Cfg.subdivisions
                loss_loc = sum(losses_loc)
                loss_conf = sum(losses_conf)
                loss_cls = sum(losses_cls)
                loss.backward()
                waste_time = time.time() - start_time
                total_loss += loss
                cur_step += 1
                #将第五个Epoch开始写入到tensorboard，每一步都写
                if epoch > 2:
                    writer.add_scalar('total_loss/gpu_batch', loss*Cfg.subdivisions, (epoch * epoch_size + iteration))
                    writer.add_scalar('loss_loc/gpu_batch', loss_loc, (epoch * epoch_size + iteration))
                    writer.add_scalar('loss_conf/gpu_batch', loss_conf, (epoch * epoch_size + iteration))
                    writer.add_scalar('loss_cls/gpu_batch', loss_cls, (epoch * epoch_size + iteration))

                if cur_step % Cfg.subdivisions == 0:
                    optimizer.step()
                    if Cfg.burn_in > 0:
                        lr_scheduler.step()
                    model.zero_grad()

                pbar.set_postfix(**{'loss_cur': loss.item()*Cfg.subdivisions,
                                    'loss_total': total_loss.item() / (iteration + 1)*Cfg.subdivisions,
                                    'lr': get_lr(optimizer),
                                    'step/s': waste_time})
                pbar.update(1)
                start_time = time.time()

        if Cfg.burn_in == 0:
            lr_scheduler.step()

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f || Last Loss: %.4f ' % (total_loss / (epoch_size + 1)*Cfg.subdivisions, loss.item()*Cfg.subdivisions))
        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(), '%s/Epoch_%03d_Loss_%.4f.pth' % (Cfg.check,
        (epoch + 1), total_loss / (epoch_size + 1)*Cfg.subdivisions))


if __name__ == "__main__":
    # 一般为608
    input_shape = (Cfg.h, Cfg.w)
    # 是否使用余弦学习率
    Cosine_lr = Cfg.cosine_lr
    # 是否使用马赛克数据增强
    mosaic = Cfg.mosaic
    # 用于设定是否使用cuda
    Cuda = True
    smoooth_label = Cfg.smoooth_label
    # -------------------------------#
    #   Dataloder的使用
    # -------------------------------#
    Use_Data_Loader = Cfg.use_data_loader
    train_data = Cfg.train_data
    # -------------------------------#
    #   获得先验框和类
    # -------------------------------#
    class_names = get_classes(Cfg.classes_path)
    num_classes = len(class_names)
    print('classes:', class_names, num_classes)

    lr = Cfg.learning_rate
    batch_size = Cfg.batch

    #是否为首次训练
    if Cfg.first_train:
        cur_epoch = 0
    else:
        cur_epoch = get_epoch_by_name(Cfg.pth_path)

    total_epoch = Cfg.total_epoch
    # 一次送入GPU的数据量
    gpu_batch = Cfg.batch // Cfg.subdivisions
    lines, num_train, num_val = get_train_lines(train_data)
    # 当前的训练batch数,用于调节是否burn_in，以及学习率，恢复训练时会使用到
    # 首次训练为0
    cur_batch = num_train * cur_epoch // batch_size
    # 1.需要生成的先验框尺寸，如果用darknet权重和cfg加载，会使用yolov4.cfg里的anchors
    # 2.对于计算训练损失，不论是darknet权重加载还是pth加载，都需要使用这个参数
    anchors = get_anchors(Cfg.anchors_path)

    pth_path = Cfg.pth_path

    if Cfg.valid_mode:
        valid([50], classes={0: 'car', 1: 'pedestrian'}, draw=Cfg.draw_box, anchors=anchors)
    else:
        train(cur_epoch, total_epoch, cuda=True, anchors=anchors)
