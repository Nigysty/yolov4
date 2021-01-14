

'''
Attention:
    The input had been written at last, two txt file should be prepared first.
    -- detpath : txt file contain detection boxes info, structure be like:
        -class1.txt
            imgname1 confidence1 box1
            imgname1 confidence2 box2
            imgname2 confidece ... 
        -class2.txt
        -class3.txt 
        (to convert json file to this, use "json2imgcls.py")
    -- imagesetfile : the testset image list
    
    Except this, you should have:
    -- /images : testset images dir
    -- /annotations: annotations' dir of the testset
    -- ovthresh : the threshold of IOU

    Outputs are recall, prediction and ap. 
        each time you can only choose one class and test the relative rec, pred and ap

'''



import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import json
import shutil

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects


def parse_json(jsonfile, classname):
    '''
    Parse a json file [YOLO2 output result]
    '''
    det_ret = json.load(file(jsonfile))
    det_bb = []
    det_conf = []

    for i in det_ret:
        topleft = i[u'topleft']
        bottomright = i[u'bottomright']
        label = i[u'label']
        if label == classname:
            conf = i[u'confidence']
            dxmin = int(topleft[u'x'])
            dymin = int(topleft[u'y'])
            dxmax = int(bottomright[u'x'])
            dymax = int(bottomright[u'y'])
            dbb = [dxmin, dymin, dxmax, dymax]
            det_bb.append(dbb)
            det_conf.append(conf)
        else:
            continue

    if det_conf == []:
        det_bb.append([0.,0.,0.,0.])
        det_conf.append(0.)

    return det_bb, det_conf




def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def voc_eval(detpath,
             annopath,
             imagepath,
             classname,
             ovthresh,
             confthresh,
             use_07_metric=False):
    
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagedirfile is a text file with each line an image dir

    # first load gt

    # read list of images
    
    xml_paths = os.listdir(annopath)
    imagenames = [x[:-4] for x in xml_paths] 
    
    recs = {}
    for i, imagename in enumerate(imagenames):
        # recs[imagename] = parse_rec(annopath.format(imagename))
        xml_path = os.path.join(annopath, imagename+'.xml')
        recs[imagename] = parse_rec(xml_path)
        if i % 100 == 0:
            print ('Reading annotation for {:d}/{:d}'.format(
                i + 1, len(imagenames)))
    
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        for i in range(len(R)):
            area = (bbox[i, 2] - bbox[i, 0] + 1.) * (bbox[i, 3] - bbox[i, 1] + 1.)
            '''
            if area >= 20*20 and difficult[i] ==False:
                npos += 1
            '''
            npos +=1
        #npos = npos + sum(~difficult) 
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    # detfile = detpath.format(classname)
    detfile = detpath
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    if confidence.shape[0] == 0:
      return 0, 0, 0
    
    if BB.shape[0] > 0:
        sorted_inds = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        for i in range(len(sorted_scores)):
            if sorted_scores[i] > -confthresh:
                valid_ind = i
                break
            else:
                valid_ind = len(sorted_scores) - 1
        BB = BB[sorted_inds[:valid_ind], :]
        image_ids = [image_ids[x] for x in sorted_inds[:valid_ind]]
    #print(len(image_ids),image_ids) 
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        #print(image_ids[d])
        #print(bb)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
            
            
            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            gtbox = (BBGT[jmax, 2] - BBGT[jmax, 0] + 1.) * (BBGT[jmax, 3] - BBGT[jmax, 1] + 1.)
            '''
            
            gtbox = (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.)
            overlaps = inters / gtbox
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            '''
        if ovmax > ovthresh:
            
            if R['difficult'][jmax]==False and gtbox >= 20*20:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
            
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    #print nd
    print ('npos {:d}'.format(npos))
    print ('nd {:d}'.format(nd))
    if len(fp)>0: 
        #print fp[-1]
        print ('fp {:d}'.format(int(fp[-1])))
    if len(tp)>0: 
        #print tp[-1]
        print ('tp {:d}'.format(int(tp[-1])))
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    #ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec


def draw_detect(label,
             detpath,
             gtpath,
             imglist,
             imagepath,
             writepath,
             ):
    import cv2
    from visualization import visualize_dets

    # read list of images
    dets = {}
    gts = {}
    gtfile = gtpath
    #imagenames = [os.path.basename(x)[:-4] for x in imglist]
    
    imagenames = []
    # read anns
    with open(gtfile,'r') as f:
        lines = f.readlines()
    for line in lines:
        imgid, _, cate, conf, bbox = line.strip().split('\t')
        
        if cate!='lo':
            continue
        imagenames.append(imgid[:-4])
        bbox = eval(bbox.strip())
        if imgid[:-4] in gts:
            gts[imgid[:-4]].append([cate,bbox])
        else:
            gts[imgid[:-4]] = []
            gts[imgid[:-4]].append([cate,bbox])
        
    # read dets
    detfile = detpath
    with open(detfile, 'r') as f:
        lines = f.readlines()
    
    image_ids=[]
    bboxs=[]
    confidence=[]
    for line in lines:
        imgid, _, cate, conf, bbox = line.strip().split('\t')
        bbox = eval(bbox.strip())
        if cate!='lo':
            continue
        if not imgid[:-4] in imagenames:
            imagenames.append(imgid[:-4])
        if imgid[:-4] in dets:
            dets[imgid[:-4]].append([cate,conf,bbox])
        else:
            dets[imgid[:-4]] = []
            dets[imgid[:-4]].append([cate,conf,bbox])
    print('<<total imagenames',len(imagenames))        
    for imagename in imagenames:
        img = cv2.imread(os.path.join(imagepath,imagename+'.jpg'))
        #print(os.path.join(imagepath,imagename+'.jpg'))
        if imagename in gts:
            gtinfs = gts[imagename]
        else:
            gtinfs = []
        if imagename in dets:
            detinfs = dets[imagename]
        else:
            detinfs = []  
        for gtinf in gtinfs:
            cate,[xmin,ymin,xmax,ymax] = gtinf
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax),int(ymax)), (0,0,255), 2)
            img = cv2.putText(img, cate, ((int(xmin), int(ymin))), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
        for index,detinf in enumerate(detinfs):
            cate,conf,[xmin,ymin,xmax,ymax] = detinf
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)
            img = cv2.putText(img, str(index), ((int(xmin), int(ymin))), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
            img = cv2.putText(img, str(index)+':'+cate+str(conf),((10, 10+index*30)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.imwrite(os.path.join(writepath, imagename+'.jpg'),img)


def voc_eval_writeimages(label,
            detpath,
             gtpath,
             imglist,
             writepath,
             ovthresh,
             confthresh,
             use_07_metric=False):
    
    import cv2
    from visualization import visualize_dets

    # read list of images
    #xml_paths = os.listdir(annopath)
    #xml_paths = [x.replace('JPEGImages','Annotations').replace('.jpg','.xml') for x in imglist]
    gtfile = gtpath
    imagenames = [os.path.basename(x)[:-4] for x in imglist]
    recs = {}
    print('<<images nums:',len(imagenames))
    
    
    anno_num = 0
    detect_num = 0
    with open(gtfile,'r') as f:
        lines = f.readlines()
    for line in lines:
        imgid, _, cate, conf, bbox = line.strip().split('\t')
        if not cate == label:
            continue
        #imagenames.append(imgid[:-4])
        bbox = eval(bbox.strip())
        anno_num+=1
        if imgid[:-4] in recs:
            recs[imgid[:-4]].append(bbox)
        else:
            recs[imgid[:-4]] = []
            recs[imgid[:-4]].append(bbox)
        
    
    print('<<label:{},anno nums:{}'.format(label,anno_num))
    '''
    for i, imagename in enumerate(imagenames):
        # recs[imagename] = parse_rec(annopath.format(imagename))
        xml_path = xml_paths[i]
        recs[imagename] = parse_rec(xml_path)
        
        #if i % 100 == 0:
            #print ('Reading annotation for {:d}/{:d}'.format(
             #   i + 1, len(imagenames)))
    '''
    #extract gt objects for this class
    #class_recs = {}
    npos = 0
    
    # read dets
    dets = {}
    detfile = detpath
    with open(detfile, 'r') as f:
        lines = f.readlines()
    
    
    image_ids=[]
    bboxs=[]
    confidence=[]
    for line in lines:
        imgid, _, cate, conf, bbox = line.strip().split('\t')
        if not cate == label:
            continue
        image_ids.append(imgid[:-4])
        confidence.append(float(conf))
        bbox = eval(bbox.strip())
        bboxs.append(bbox)
        detect_num+=1
        
    print('<<label:{},detect nums:{}'.format(label,detect_num))
    
    
    confidence = np.array(confidence)
    BB = np.array(bboxs)
    pre_image_id = image_ids[0]
    id = 0
    for i in range(len(image_ids)):
        image_id = image_ids[i]
        if image_id != pre_image_id: 
            dets[pre_image_id] ={'bbox': BB[id:i,:],
                                 'confidence': confidence[id:i]}
            id = i
            pre_image_id = image_id
        if i == len(lines)-1:
            dets[pre_image_id] ={'bbox': BB[id:i,:],
                                 'confidence': confidence[id:i]}
    if confidence.shape[0] == 0:
      return 0
    
    nd = len(image_ids)
    tp = 0
    tn = 0 
    fp = 0
    valid = 0
    npos = 0
    list_wrong = []
    for index,imagename in enumerate(imagenames):
        try:
            bbox = recs[imagename]
        except:
            continue
        #difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        #det = [False] * len(R)
        #npos = npos + sum(~difficult) +sum(difficult)
        if len(bbox)==0 and imagename not in dets:
            continue
        if imagename not in dets:
            
            tn += len(bbox)
            list_wrong.append(imglist[index])
            
            continue
        det = dets[imagename]
        bb = det['bbox']
        confidence = det['confidence']
        #print(len(bb[:,0]))
        if bb.shape[0] > 0:
            sorted_inds = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            for i in range(len(sorted_scores)):
                if sorted_scores[i] > -confthresh:  #-0.000001
                    valid_ind = i
                    break
                else:
                    valid_ind = len(sorted_scores)
            bb = bb[sorted_inds[:valid_ind], :]
        #print(len(bb[:,0]))
        #im = cv2.cvtColor(cv2.imread(os.path.join(imagepath,imagename + '.jpg')), cv2.COLOR_BGR2RGB)
        #visualize_dets(im, bb, bbox, 1.0, save_path=os.path.join(writepath, imagename + '.jpg'))
        if(len(bb[:,0]) > 0):
            match_gt = 0
            for BBGT in bbox:
                BBGT = np.array(BBGT)
                if BBGT.size > 0:
                # compute overlaps
                # intersection
                    ixmin = np.maximum(bb[:, 0], BBGT[0])
                    iymin = np.maximum(bb[:, 1], BBGT[1])
                    ixmax = np.minimum(bb[:, 2], BBGT[2])
                    iymax = np.minimum(bb[:, 3], BBGT[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih 
                    gtbox = (BBGT[2] - BBGT[0] + 1.) * (BBGT[3] - BBGT[1] + 1.)
                    
                    
                    uni = ((bb[:,2] - bb[:,0] + 1.) * (bb[:,3] - bb[:,1] + 1.) + gtbox - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    #jmax = np.argmax(overlaps)
                    
                    if ovmax > ovthresh:
                        tp += 1
                        match_gt +=1
                        #im = cv2.cvtColor(cv2.imread(os.path.join(imagepath,imagename + '.jpg')), cv2.COLOR_BGR2RGB)
                        #visualize_dets(im, bb, bbox, 1.0, save_path=os.path.join(writepath, imagename + '.jpg'))
                    else:
                        tn += 1
                        list_wrong.append(imglist[index])
                        #im = cv2.cvtColor(cv2.imread(os.path.join(imagepath,imagename + '.jpg')), cv2.COLOR_BGR2RGB)
                        #visualize_dets(im, bb,bbox, 1.0, save_path=os.path.join(writepath, 'tn', imagename + '.jpg'))
            if len(bb[:,0]) >  match_gt:
                fp += len(bb[:,0]) - match_gt
                list_wrong.append(imglist[index])
                #im = cv2.cvtColor(cv2.imread(os.path.join(imagepath,imagename + '.jpg')), cv2.COLOR_BGR2RGB)
                #visualize_dets(im, bb,bbox, 1.0, save_path=os.path.join(writepath, 'fp', imagename + '.jpg'))
            if len(bb[:,0]) <  match_gt:
                print('something wrong')
            #imwrite proposal in images 
            #print ('write images' + imagename)
            #im = cv2.cvtColor(cv2.imread(os.path.join(imagepath,imagename + '.jpg')), cv2.COLOR_BGR2RGB)
            #visualize_dets(im, bb, bbox, 1.0, save_path=os.path.join(writepath, imagename + '.jpg'))
        else:
            if(len(bbox)>0):
                tn +=len(bbox)
                list_wrong.append(imglist[index])
                valid += 1
                #print(bb)
                #print(len(bbox),bbox)
                #im = cv2.cvtColor(cv2.imread(os.path.join(imagepath,imagename + '.jpg')), cv2.COLOR_BGR2RGB)
                #visualize_dets(im, bb,bbox, 1.0, save_path=os.path.join(writepath, 'tnn', imagename + '.jpg'))
    # compute precision recall
    #fp = np.cumsum(fp)
    rec = float(tp) / float(tp + tn)
    prec = float(tp) / float(tp + fp)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    #prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    #ap = voc_ap(rec, prec, use_07_metric)
    print ('tn {:d}'.format(tn))
    
    print ('tp {:d}'.format(tp)) 
    print ('npos {:d}'.format(npos))
    print ("recall: ", rec)
    print  ('prec:',prec)
    print(len(list_wrong))
    
    for image_path in list_wrong:
        #print(image_name)
        #print(recs['+nfs232+nfsc_oculi+oculi+process+splite+2019_01_15+20190109-027VG-195506+20190109144046576+20190109144046576=70'])
        
        image_name = os.path.basename(image_path)[:-4]
        #R = [obj for obj in recs[image_name]]
        #bbox = np.array([x['bbox'] for x in R])
        bbox = recs[image_name]
        #print(R)
        if image_name in dets:
            det = dets[image_name]
            bb = det['bbox']
            confidence = det['confidence']
            #bb = [bb[x] for x in range(len(confidence)) if confidence[x] > confthresh]
        else:
            bb = []
        img = cv2.imread(image_path)
        for box in bbox:
            xmin,ymin,xmax,ymax = box
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax),int(ymax)), (0,0,255), 2)
        for index,box in enumerate(bb):
            xmin,ymin,xmax,ymax = box
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)
            img = cv2.putText(img, str(confidence[index]), ((10, 10+index*30)), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.imwrite(os.path.join(writepath, image_name+'.jpg'),img) 
        '''
        out_img_file = os.path.join(writepath,image_path[len('/data/traffic_sign_10_classes_date0820_fix_panel/'):])
        out_xml_file = out_img_file.replace('JPEGImages','Annotations').replace('.jpg','.xml')
        out_img_path = os.path.dirname(out_img_file)
        out_xml_path = os.path.dirname(out_xml_file)
        if not os.path.exists(out_img_path):
            os.makedirs(out_img_path)
        if not os.path.exists(out_xml_path):
            os.makedirs(out_xml_path)
        xml_in_path = '/data/traffic_sign_10_classes_date0820_fix_panel/20190416/Annotations/'
        shutil.copy(image_path,out_img_file)
        shutil.copy(image_path.replace('JPEGImages','Annotations').replace('.jpg','.xml'),out_xml_file)
        '''
    
if __name__ == '__main__':
    
    result_txt_path = './results/20191028_9class.txt'
    gtbox_txt_path = './gtbox/20191028_9class.txt'
    imagepath = '/data/2019-10-28/images/'
    #result_txt_path = 'yolo3-tiny/gisdata_prohibit.txt'
    label = 'lo'
    writepath = 'images_out/{}'.format(label)
    
    '''
    list_file = '20190416.lists'
    
    with open(list_file,'r') as f:
        image_list = f.readlines()
    
    imglist = [x.strip() for x in image_list] 
    '''
    img_names = os.listdir(imagepath)
    imglist = [os.path.join(imagepath,x) for x in img_names]
    
    #print('{}'.format(image_list[0]))
    
    if not os.path.exists(writepath):
        os.mkdir(writepath)
    
    draw_detect(label = label,
            detpath = result_txt_path,
            gtpath = gtbox_txt_path,
            imagepath = imagepath,
            imglist = imglist,
            writepath = writepath)
    '''
    rec, prec = voc_eval(
            detpath = result_txt_path,
            annopath = annopath,
            imagepath = imagepath,
            classname = classname,
            ovthresh = 0.5,
            confthresh = 0.3,
            use_07_metric=False)
    print ("recall: ", rec[-1])
    print ("prec: ", prec[-1])          
    
    
    voc_eval_writeimages(
            label = label,
            detpath = result_txt_path,
            gtpath = gtbox_txt_path,
            imglist = imglist,
            writepath = writepath,
            ovthresh=0.4,
            confthresh = 0.1,
            use_07_metric=False)
    '''
    