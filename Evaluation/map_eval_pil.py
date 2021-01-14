
import numpy as np
import argparse
import os, sys, time, ast
from PIL import Image, ImageDraw, ImageFont



def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua    



def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

class map_eval:
    def __init__(self,detect_file,anno_file,flag_subclass):
        self.anno_file = anno_file
        self.detect_file = detect_file
        self.dictionary = []
        self.image_list = []
        annots = self.get_info_from_file(self.anno_file,flag_subclass,True)
        dects = self.get_info_from_file(self.detect_file,flag_subclass,False)
        self.dictionary.sort()
        print("Dictionary:\t", self.dictionary)
        print("ImageSet:  \t", len(self.image_list))
        self.all_annotations = self._get_all_info(annots)
        self.all_detections = self._get_all_info(dects)
        #miss box and false box
        self.missboxs = {}
        self.falseboxs = {}

    def get_info_from_file(self,file,flag_subclass,flag_size_filter):
        lines = open(file, 'r').readlines()
        outs = []
        for i,line in enumerate(lines):
            imgid, _, cate, conf, bbox = line.strip().split('\t')
            imgid=imgid.strip()
            cate = cate.strip()
            conf = conf.strip()
            bbox = eval(bbox.strip())
            
            
            width = bbox[2]-bbox[0]
            height = bbox[3]-bbox[1]
            
            ###size filter
            # if flag_size_filter:
            #     if width<30 or height<30:
            #         continue
            ###
            #print("{}temp --> {}".format(i,line))
            # if not flag_subclass:
            #     ###
            #     if cate =='panel':
            #         cate = 'panel'
            #     elif cate[0]=='person':
            #         cate = 'po'
            #     elif cate[0]=='i':
            #         cate = 'io'
            #     elif cate[0]=='w':
            #         cate = 'wo'
            #
            #     elif cate in ['tl','rl','gl','sgl','lgl','rgl'] or cate[-2:]=='tl':
            #         cate = 'tl'
            #     ###
            #
            # else:
            #     ###
            #     if cate[-2:]=='tl' and len(cate)>2:
            #         if cate[-3]=='r':
            #             cate = 'rl'
            #         elif cate[-3]=='g':
            #             cate = 'gl'
            #         else:
            #             cate = 'tl'
            #
            #     if not (cate in ['pg','pn','pne','ps','p5','p9','p10','p14','p11','p19','p20','p23','p26'] or cate[:2]=='pl'):
            #         continue
                ###
            if imgid not in self.image_list:
                self.image_list.append(imgid)
            if cate not in self.dictionary:
                self.dictionary.append(cate)
            outs.append([imgid, cate, conf, bbox])
        return outs
    
    def _get_all_info(self,infos):
        # print("Start _get_annotations() ......")
        all_info = [[ [] for i in range(len(self.dictionary))] for j in range(len(self.image_list))]

        for p in range(len(infos)):
            img_name, _cate, _conf, _box = infos[p]
            i = self.image_list.index(img_name)
            _label = self.dictionary.index(_cate)
            all_info[i][_label].append([_box,_conf])                
        return all_info
    
    def evaluate(self,confidence,iou_threshold=0.5):        
        recalls = {}; precisions = {}; average_precisions = {}; num_lib = {}

        for label in range(len(self.dictionary)):
            label_name = self.dictionary[label]
            false_positives, true_positives, scores, num_annotations, num_detections, num_hit,missbox,falsebox = \
                self.box_relations_via_labels(label=label,conf_thresh=confidence, iou_threshold=iou_threshold)

            num_lib[label_name] = [num_annotations, num_detections, num_hit]   ## format [#anno, #dect, #hit] 
            recall, precision, average_precision = measurements_calculator(false_positives, true_positives, scores, num_annotations)

            recalls[label_name] = recall
            precisions[label_name] = precision
            average_precisions[label_name] = average_precision
            
            self.missboxs[label_name] = missbox
            self.falseboxs[label_name] = falsebox

        return recalls, precisions, average_precisions, num_lib
    
    def box_relations_via_labels(self,label,conf_thresh,iou_threshold=0.5):
        '''
        missbox_recorder   : to record missbox images, `missbox_recorder` object must be an object with attribute `write`
        '''
        label_name = self.dictionary[label]
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        num_detections  = 0.0
        num_hit         = 0.0
        missbox = []
        falsebox = []
        
        for i in range(len(self.image_list)):
            _detections                  = self.all_detections[i][label]
            _annotations                 = self.all_annotations[i][label]
            _annotations = [x[0] for x in _annotations]
            ## confidence filter
            _detections                  = confd_filter(_detections, conf_thresh) 
            detections                   = np.array(_detections)
            annotations                  = np.array(_annotations)
            num_annotations              += annotations.shape[0]
            num_detections               += detections.shape[0]
            detected_annotations         = []

            ### evaluation measurements calculator ###
            
            if len(_detections)==0:
                for x in _annotations:
                    missbox.append([self.image_list[i],x])
                continue
            for d_id, d in enumerate(detections):
                conf = float(d[1])
                _d = []
                for numb in d[0]:
                    ## xmin,ymin,xmax,ymax
                    _d.append(float(numb))
                xmin,ymin,xmax,ymax = _d
                width = int(xmax) - int(xmin)
                height = int(ymax) - int(ymin)
                ## confidence
                _d.append(float(d[1]))
                d = _d

                if annotations.shape[0] == 0:
                    # if width<30 or height<30:
                    #     num_detections -=1
                    #     continue
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    falsebox.append([self.image_list[i],d])
                    scores = np.append(scores, conf)
                    continue

                overlaps            = compute_overlap(np.expand_dims(np.array(d), axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                    num_hit += 1
                    scores = np.append(scores, conf)
                else:
                    # if width<30 or height<30:
                    #     num_detections-=1
                    #     continue
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    falsebox.append([self.image_list[i],d])
                    scores = np.append(scores, conf)
            for anno_idx in range(len(_annotations)):
                if anno_idx not in detected_annotations:
                    missbox.append([self.image_list[i],_annotations[anno_idx]])               
        return false_positives, true_positives, scores, num_annotations, num_detections, num_hit,missbox,falsebox
    
    def recorder_miss_flase_box(self,out_dir):
        
        for label_name in self.dictionary:
            if not os.path.exists(os.path.join(out_dir,label_name)):
                os.makedirs(os.path.join(out_dir,label_name))
            missbox = self.missboxs[label_name]
            falsebox = self.falseboxs[label_name]
            with open(os.path.join(out_dir,label_name,'{}_missbox.txt'.format(label_name)),'w') as f:
                for line in missbox:
                    f.write('{}\n'.format(line))
            with open(os.path.join(out_dir,label_name,'{}_falsebox.txt'.format(label_name)),'w') as f:
                for line in falsebox:
                    f.write('{}\n'.format(line))
    
    def draw_full_img(self,out_dir,image_dir):
        for label_name in self.dictionary:
            label_index = self.dictionary.index(label_name)
                     
            missbox = self.missboxs[label_name]
            falsebox = self.falseboxs[label_name]
            miss_names = [x[0] for x in missbox]
            false_names = [x[0] for x in falsebox]
            
            font = ImageFont.truetype("../ttf/fangzheng_fangsong.ttf",15)
            fontcolor = "#0000ff"  #
            label_path = os.path.join(out_dir,label_name)
            if not os.path.exists(label_path):
                os.makedirs(label_path)
            for img_name in miss_names:
                img_index = self.image_list.index(img_name)
                annos = self.all_annotations[img_index][label_index]
                dets = self.all_detections[img_index][label_index]
                
                img = Image.open(os.path.join(image_dir,img_name))
                width,height = img.size[:2]
                draw = ImageDraw.Draw(img)
                writepath = os.path.join(label_path,'missbox','full_image')
                if not os.path.exists(writepath):
                    os.makedirs(writepath)
                for box in annos:
                    xmin,ymin,xmax,ymax = border_process(width,height,box[0])
                    draw.rectangle((xmin,ymin,xmax,ymax),outline="#0000ff")
                for index,box in enumerate(dets):
                    conf= box[1]
                    xmin,ymin,xmax,ymax = border_process(width,height,box[0])
                    info = str(index) + ": "+ conf
                    draw.rectangle((xmin,ymin,xmax,ymax),outline="#00ff00")
                    draw.text((int(xmin) - 10, int(ymin) - 10),str(index),font=font,fill=fontcolor)
                    draw.text((10, int(10 + 20 * index)),info,font=font,fill=fontcolor)
                img.save(os.path.join(writepath, img_name))
            
            for img_name in false_names:
                img_index = self.image_list.index(img_name)
                annos = self.all_annotations[img_index][label_index]
                dets = self.all_detections[img_index][label_index]

                img = Image.open(os.path.join(image_dir,img_name))
                width,height = img.size[:2]
                draw = ImageDraw.Draw(img)
                writepath = os.path.join(label_path,'falsebox','full_image')
                if not os.path.exists(writepath):
                    os.makedirs(writepath)
                for box in annos:
                    xmin,ymin,xmax,ymax = border_process(width,height,box[0])
                    draw.rectangle((xmin,ymin,xmax,ymax),outline="#0000ff")
                for index,box in enumerate(dets):
                    conf= box[1]
                    xmin,ymin,xmax,ymax = border_process(width,height,box[0])
                    info = str(index) + ": "+ conf
                    draw.rectangle((xmin,ymin,xmax,ymax),outline="#00ff00")
                    draw.text((int(xmin) - 10, int(ymin) - 10),str(index),font=font,fill=fontcolor)
                    draw.text((10, int(10 + 20 * index)),info,font=font,fill=fontcolor)
                img.save(os.path.join(writepath, img_name))
    
    def draw_box(self,out_dir,image_dir,flag_cut,flag_full):
        for label_name in self.dictionary:
            label_index = self.dictionary.index(label_name)
                     
            missbox = self.missboxs[label_name]
            falsebox = self.falseboxs[label_name]
            miss_names = [x[0] for x in missbox]
            false_names = [x[0] for x in falsebox]
            
            label_path = os.path.join(out_dir,label_name)
            if not os.path.exists(label_path):
                os.makedirs(label_path)
            for i,img_name in enumerate(miss_names):
                img = Image.open(os.path.join(image_dir,img_name))
                width,height = img.size[:2]
                box = border_process(width,height,missbox[i][1])
                xmin,ymin,xmax,ymax = box
                if box[0]>=box[2] or box[1]>=box[3]:
                    continue
                if flag_cut:
                    writepath = os.path.join(label_path,'missbox','cut_piece')
                    if not os.path.exists(writepath):
                        os.makedirs(writepath)
                    img_cut = img.crop((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
                    img_cut.save(os.path.join(writepath, img_name[:-4]+'_{}.jpg'.format(i)))
                if flag_full:
                    draw = ImageDraw.Draw(img)
                    writepath = os.path.join(label_path,'missbox','full_image')
                    if not os.path.exists(writepath):
                        os.makedirs(writepath)
                    draw.rectangle((xmin,ymin,xmax,ymax),outline="#ff0000")    
                    img.save(os.path.join(writepath, img_name[:-4]+'_{}.jpg'.format(i)))
  
            for i,img_name in enumerate(false_names):
                img = Image.open(os.path.join(image_dir,img_name))
                width,height = img.size[:2]
                box = border_process(width,height,falsebox[i][1][:4])
                
                xmin,ymin,xmax,ymax = box
                if box[0]>=box[2] or box[1]>=box[3]:
                    continue
                if flag_cut:
                    writepath = os.path.join(label_path,'falsebox','cut_piece') 
                    if not os.path.exists(writepath):
                        os.makedirs(writepath)
                    img_cut = img.crop((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
                    img_cut.save(os.path.join(writepath, img_name[:-4]+'_{}.jpg'.format(i)))
                if flag_full:
                    draw = ImageDraw.Draw(img)
                    writepath = os.path.join(label_path,'falsebox','full_image')
                    if not os.path.exists(writepath):
                        os.makedirs(writepath)
                    draw.rectangle((xmin,ymin,xmax,ymax),outline="#ff0000")    
                    img.save(os.path.join(writepath, img_name[:-4]+'_{}.jpg'.format(i)))    

def border_process(width,height,box):
    try:
        xmin,ymin,xmax,ymax = box
        xmin = max(float(xmin),0.)
        ymin = max(float(ymin),0.)
        xmax = min(float(xmax),float(width-1))
        ymax = min(float(ymax),float(height-1))
    except Exception as e:
        print('<<<<<<<<<<',box, e)
    return [xmin,ymin,xmax,ymax]

    
def scale_filter(boxset, tagset, scale_id):
    newboxset = []
    if len(boxset) != len(tagset):
        raise 'Error: boxset cannot match tagset'
    for box_id, box in enumerate(boxset):
        tag = tagset[box_id]
        if tag != scale_id:
            continue
        else:
            newboxset.append(box)
    return newboxset


def confd_filter(boxset, conf_thresh):
    outboxs = []
    outtags = []
    if len(boxset) == 0:
        return boxset
    for idx, objects in enumerate(boxset):
        _box, _score = objects
        if float(_score) >= conf_thresh:
            outboxs.append(objects)
    return outboxs

def measurements_calculator(false_positives, true_positives, scores, num_annotations):
    # no annotations -> AP for this class is 0 (is this correct?)
    if num_annotations == 0:
        recall = 'X'
        precision = 'X'
        average_precision = 0
        return recall, precision, average_precision

    # sort by score
    indices         = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives  = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)

    # compute recall and precision
    recall    = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    # print("precision calculate", true_positives, np.maximum(true_positives + false_positives, np.finfo(np.float64).eps))

    # compute average precision
    average_precision  = _compute_ap(recall, precision)

    return recall, precision, average_precision

def output_controller(recalls, precisions, average_precisions, num_lib, dictionary, cal_dect=False, cal_scale=False):
    '''
    by changing the parameter `dictionary`, you can choose to calculate the AP of specific set of classes
    '''
    tag = 0.
    sumup = 0.
    total_annos = 0.
    total_dects = 0.
    total_hits  = 0.

    for label_name in dictionary:
        if len(recalls[label_name]) !=0 :
            rcl = recalls[label_name][-1]
            if isinstance(rcl, float):
                rcl = round(rcl, 4)
        else:
            rcl = 0.
        if len(precisions[label_name]) !=0 :
            prc = precisions[label_name][-1]
            if isinstance(prc, float):
                prc = round(prc, 4)
        else:
            prc = 0.

        apr = round(average_precisions[label_name], 4)
        num_anno, num_dect, num_hit = num_lib[label_name]
        num_anno = int(num_anno); 
        num_dect = int(num_dect); 
        num_hit = int(num_hit); 

        total_annos += num_anno
        total_dects += num_dect
        total_hits  += num_hit             

        if not cal_dect:
            print('{0}\t{1}:{2:<5}\t{3}:{4:<5}\t{5}:{6:<5}\t\t{7}:{8:<5}\t{9}:{10:<5}\t{11}:{12:<5}'.format(label_name, '#Annot', str(num_anno), \
                '#Dects', str(num_dect), '#Hit', str(num_hit), '#AP', str(apr), '#Precision', str(prc), '#Recall', str(rcl)))        
        if prc != 'X' :
            tag += 1
            sumup += apr

    ### Result Summary
    if not cal_dect:
        # print('\n')
        # print(    'tag (have gt-boxes class):', tag)
        if tag > 0:
            print('Sum #Anno:{}\tSum #Dects:{}\tSum #Hits:{}\t'.format(str(int(total_annos)), str(int(total_dects)), str(int(total_hits))))
            print('Total Precision:{}\tTotal Recall:{}\t'.format(str(round(total_hits/(total_dects+0.01), 3)), str(round(total_hits/total_annos, 3))))
            if not cal_scale:
                print('Whole      mAP [{} type]:\t{}'.format(str(len(dictionary)), str(round(sumup/len(average_precisions),3))))
                # print('Non-sparse mAP [{} type]:\t{}'.format(str(int(tag)), str(round(sumup/tag,3))))
        else:
            print('No ground-truth box. Skip it!')
        print('===========================================================================================================')
    else:
        if tag > 0:
            print('Sum #Anno:{}\tSum #Dects:{}\tSum #Hits:{}\t'.format(str(int(total_annos)), str(int(total_dects)), str(int(total_hits))))
            print('Total Precision:{}\tTotal Recall:{}\t'.format(str(round(total_hits/(total_dects+0.01), 3)), str(round(total_hits/total_annos, 3))))
            if not cal_scale:
                print('Whole      mAP [{} type]:\t{}'.format(str(len(dictionary)), str(round(sumup/len(average_precisions),3))))
                print('Non-sparse mAP [{} type]:\t{}'.format(str(int(tag)), str(round(sumup/tag,3))))
        else:
            print('No ground-truth box. Skip it!')
        print('==========================================================================================================='        )

    Map = round(sumup/len(average_precisions)+1e-7, 3)
    return Map

from easydict import EasyDict

def compute_map(args):
    process = map_eval(args.detection_file, args.annotation_file, args.detect_subclass)

    # print( \
    #     '   Parameters: \n \
    #         annotation_file:   {} \n \
    #         detection_file:    {} \n \
    #         confidence:        {} \n \
    #         iou:               {} \n \
    #         out_dir:               {}'.format(args.annotation_file, args.detection_file, args.confidence, args.iou,
    #                                           args.out_dir))

    recalls, precisions, average_precisions, num_lib = process.evaluate(args.confidence, args.iou)
    Map = output_controller(recalls=recalls, precisions=precisions, average_precisions=average_precisions, \
                      num_lib=num_lib, dictionary=process.dictionary, cal_dect=False, cal_scale=False)

    if args.record_mistake:
        process.recorder_miss_flase_box(args.out_dir)

    if args.draw_cut_box or args.draw_full_img:
        process.draw_box(args.out_dir, args.input_dir, args.draw_cut_box, args.draw_full_img)

    return Map



if __name__ == '__main__':
    args = EasyDict()
    args.annotation_file = 'true.txt'
    args.detection_file = 'out.txt'
    args.detect_subclass = False
    args.confidence = 0.3
    args.iou = 0.2
    args.record_mistake = True
    args.draw_full_img = True  # True：将错检或漏检的图片保存下来，保存地址outdir
    args.draw_cut_box = False
    args.input_dir = r'/home/yolov4-pytorch1/object-detection-crowdai'
    args.out_dir = 'out_dir'
    compute_map(args)
    ### Arguments
    # parser = argparse.ArgumentParser(description='Evaluation Module -- For precision, recall and mAP calculation')
    # parser.add_argument('--annotation_file', default='true.txt',  type=str,    help='ground truth box text file (annot_file), skip if have buffer')
    # parser.add_argument('--detection_file',  default='out.txt',  type=str,    help='detection box output text file (dects_file), skip if have buffer')
    # parser.add_argument('--detect_subclass', default=False, type=ast.literal_eval,  help='if true ,do subclass,else do')
    # parser.add_argument('--confidence',      default=0.4,   type=float,  help='Confidence Threshold')
    # parser.add_argument('--iou',             default=0.3,   type=float,  help='IOU Threshold')
    # parser.add_argument('--record_mistake',  default=True, type=ast.literal_eval, help='output MISS-GTBOX and False-Detectbox recorder file')
    # parser.add_argument('--draw_full_img',   default=False, type=ast.literal_eval, help='output MISS-GTBOX and False-Detectbox images')
    # parser.add_argument('--draw_cut_box',    default=False, type=ast.literal_eval, help='output MISS-GTBOX and False-Detectbox cut box')
    # parser.add_argument('--input_dir',       default=r'D:\car_cls\yolo4_dataset_20200715\images', type=str, help='dir to read images')
    # parser.add_argument('--out_dir',         default='out_dir', type=str, help='dir to output MISS-GTBOX and False-Detectbox recorder file')
    # args = parser.parse_args()
    #
    # process = map_eval(args.detection_file,args.annotation_file,args.detect_subclass)
    #
    # print(\
    # 'Evaluation Script for detecion algorithm: \n\
    #  to change the parameter, try `python map_eval.py --help` \n\
    #     Parameters: \n \
    #     annotation_file:   {} \n \
    #     detection_file:    {} \n \
    #     confidence:        {} \n \
    #     iou:               {} \n \
    #     out_dir:               {}'.format(args.annotation_file, args.detection_file, args.confidence, args.iou, args.out_dir))
    #
    #
    # recalls, precisions, average_precisions, num_lib= process.evaluate(args.confidence,args.iou)
    # output_controller(recalls=recalls, precisions=precisions, average_precisions=average_precisions, \
    #                     num_lib=num_lib, dictionary=process.dictionary, cal_dect=False, cal_scale=False)
    #
    # if args.record_mistake:
    #     process.recorder_miss_flase_box(args.out_dir)
    #
    # if args.draw_cut_box or args.draw_full_img:
    #     process.draw_box(args.out_dir,args.input_dir,args.draw_cut_box,args.draw_full_img)
