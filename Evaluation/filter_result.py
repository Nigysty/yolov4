
import sys
import numpy as np
import os
import time


result_path = r'X:\test\temp\sign-v187-test\results.txt'
with open(result_path,'r') as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
lines = lines[1:]
out_dir = r'E:\data\testset\13-classes-test\DT-baseline-13-20191221.info'
with open(out_dir,'w') as f:
    for line in lines:
        try:
            img_name,label,conf,dlabel,dconf,rlabel,rconf,xmin,ymin,xmax,ymax = line.split('\t')
            box = [int(xmin),int(ymin),int(xmax),int(ymax)]
        except:
            continue
        
        if 'ER' in label or 'er' in label:
                continue
        elif 'W57' == label:
            label = 'w57'
        elif 'W32' ==label:
            label = 'w32'
        
        
        
        '''
        if label in ['tl','rl','gl','sgl','lgl','rgl']:
            if label in ['sgl','lgl','rgl']:
                    label = 'gl'
        '''
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(img_name,'label:',label, conf, box))
