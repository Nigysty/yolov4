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
import glob

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

## JSON version
def get_json_cord(jsonpath):
    import json
    jsonfile = json.load(open(jsonpath))
    rets = []
    outputs = jsonfile['outputs']
    if 'object' in outputs.keys():
        for obj in jsonfile['outputs']['object']:
            cate = str(obj['name'])
            xmin = int(float(obj['bndbox']['xmin']))
            ymin = int(float(obj['bndbox']['ymin']))
            xmax = int(float(obj['bndbox']['xmax']))
            ymax = int(float(obj['bndbox']['ymax']))
            if xmin >= xmax or ymin >= ymax:
                continue
            rets.append([cate, [int(xmin), int(ymin), int(xmax), int(ymax)]])
    return rets


out_dir = sys.argv[1] #'gtbox/GTBOX_GIS-SIGN-FG-Base1.txt'
annoth_path = sys.argv[2] #'/data/Traffic_Cognition/TESTSET/GIS-SIGN-FG_baseline/GIS-SIGN-FG-BASE1/labels'
xml_list = glob.glob(os.path.join(annoth_path, '*.json'))

file = open(out_dir,'w')
print('<<<<<<<<<<<<<<',annoth_path)
plo_counter = 0 
for xml_name in xml_list:
        ret = get_json_cord(xml_name)
        img_name = os.path.basename(xml_name).replace('json', 'jpg')
        for box in ret:
            label = box[0]
            
            '''
            # ## abnormal label
            # if label in ['pl', 'pl25', 'ph49', 'pa1o', 'pl80.', 'ph20', 'pa14.', 'pl20.', 'pl2o', 'pl5o']:
            #     continue
            # if label in ['pl1', 'pl20.', 'pl2o', 'plo', 'pw.2.5']:
            #     continue

             ## GISDATA-6
            if 'ER' in label or 'er' in label or label=='back':
                continue
            elif 'W57' == label:
                label = 'w57'
            elif 'W32' ==label:
                label = 'w32'
            elif label in ['lo','ors','rn','ro','tas'] or label[:2] in ['lo','ors','rn','ro']:
                label = 'panel'
                         
            ## traffic light
            if label in ['tl','rl','gl','sgl','lgl','rgl']:
                if label in ['sgl','lgl','rgl']:
                    label = 'gl'
            '''

            if label[:2] == 'pa':
                if label not in ['pa13', 'pa14']:
                    label = 'pao'
            elif label[:2] == 'ph':
                if label[:3] not in ['ph2', 'ph3', 'ph4', 'ph5']:
                    label = 'pho'
                else:
                    label = label[:3]
            elif label[:2] == 'pw':
                if label[:3] not in ['pw2', 'pw3', 'pw4']:
                    label = 'pwo'
                elif label == 'pw.2.5':
                    label = 'pw2'
                else:
                    label = label[:3]                    
            elif label[:2] == 'pm':
                if label not in ['pm10', 'pm15', 'pm20', 'pm30', 'pm40', 'pm49', 'pm55']:
                    label = 'pmo'
            elif label[:2] == 'pl':
                if label not in ['pl120', 'pl60', 'pl40', 'pl30', 'pl110', 'pl70', 'pl5', 'pl100', 'pl80', 'pl90', 'pl35', 'pl10', 'pl15', 'pl20', 'pl50']:
                    plo_counter += 1
                    continue

            file.write('{}\t{}\t{}\t{}\t{}\n'.format(img_name,'label:',label, 1.0, box[1]))
file.close()

print(">> plo_counter:", plo_counter)


