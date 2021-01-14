#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:32:38 2018

@author: horacce
"""

import sys
import numpy as np
import os
import time


def get_cord(xml_adr):
    import xml.etree.cElementTree as ET
    tree = ET.parse(xml_adr)
    root = tree.getroot()
    ret = []
    for child in root.findall("object"):
        cate = child.find("name").text
        box = child.find("bndbox")
        xmin = box.find("xmin").text
        ymin = box.find("ymin").text
        xmax = box.find("xmax").text
        ymax = box.find("ymax").text
        box = [int(xmin), int(ymin), int(xmax), int(ymax)]
        #if cate == 'prohibit':
        ret.append([cate, box])
    return ret


out_dir = 'gtbox/baseline_didi.txt'
annoth_path1 = '/data/GIS-SIGN_baseline/device_didi/Annotations'
xml_list1 = os.listdir(annoth_path1)
#xml_list1 = []

annoth_path2 = '/data/GIS-SIGN_baseline/device_didi/Annotations'
xml_list2 = os.listdir(annoth_path2)

file = open(out_dir,'w')
for xml_list,annoth_path in zip([xml_list1,xml_list2],[annoth_path1,annoth_path2]):
    print('<<<<<<<<<<<<<<',annoth_path)
    for xml_name in xml_list:
        xml_dir = os.path.join(annoth_path,xml_name)
        ret = get_cord(xml_dir)
        img_name = xml_name[:-4]+'.jpg'
        for box in ret:
            label = box[0]
            if 'ER' in label or 'er' in label or label=='back':
                continue
            elif 'W57' == label:
                label = 'w57'
            elif 'W32' ==label:
                label = 'w32'
            elif label in ['lo','ors','rn','ro','tas'] or label[:2] in ['lo','ors','rn','ro']:
                label = 'panel'
            '''
            if label in ['tl','rl','gl','sgl','lgl','rgl']:
                if label in ['sgl','lgl','rgl']:
                    label = 'gl'
            '''
            file.write('{}\t{}\t{}\t{}\t{}\n'.format(img_name,'label:',label, 1.0, box[1]))
file.close()
