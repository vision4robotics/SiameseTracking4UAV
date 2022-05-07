from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch as t
from snot.core.config_apn import cfg
from snot.utils.bbox import IoU


class AnchorTarget3_apn():
    def __init__(self):
        return

    def select(self,position, keep_num=16):
        num = position[0].shape[0]
        if num <= keep_num:
            return position, num
        slt = np.arange(num)
        np.random.shuffle(slt)
        slt = slt[:keep_num]
        return tuple(p[slt] for p in position), keep_num
    
    def get(self, anchors,targets, size):
        num=cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU
        anchor_num=1
        cls = -1 * np.ones((num,anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((num,4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((num,anchor_num, size, size), dtype=np.float32)
        overlap = np.zeros((num,anchor_num, size, size), dtype=np.float32)
        for i in range(num):
            anchor=anchors[i]
            target=targets[i].cpu().numpy()
            index=np.minimum(size-1,np.maximum(0,np.int32((target-cfg.TRAIN.MOV)/cfg.TRAIN.STRIDE)))
            w=int(index[2]-index[0])
            h=int(index[3]-index[1])
            neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
            # -1 ignore 0 negative 1 positive
            tcx = (target[0]+target[2])/2
            tcy= (target[1]+target[3])/2
            tw=target[2]-target[0]
            th=target[3]-target[1]
            if neg :
                cx = size // 2
                cy = size // 2
                cx += int(np.ceil((tcx - cfg.TRAIN.SEARCH_SIZE // 2) /
                          8 + 0.5))
                cy += int(np.ceil((tcy - cfg.TRAIN.SEARCH_SIZE // 2) /
                          8 + 0.5))
                l = max(0, cx - 3)
                r = min(size, cx + 4)
                u = max(0, cy - 3)
                d = min(size, cy + 4)
                cls[i,:, u:d,l:r ] = 0
                neg, neg_num = self.select(np.where(cls[i][0] == 0), cfg.TRAIN.NEG_NUM)
                cls[i] = -1
                cls[i][0][neg] = 0
                overlap[i] = np.zeros((anchor_num, size, size), dtype=np.float32)
                continue

            cx, cy, w, h = anchor[:,0].reshape(1,size,size),anchor[:,1].reshape(1,size,size),anchor[:,2].reshape(1,size,size),anchor[:,3].reshape(1,size,size)
            x1 = cx - w * 0.5
            y1 = cy - h * 0.5
            x2 = cx + w * 0.5
            y2 = cy + h * 0.5
            index=np.minimum(size-1,np.maximum(0,np.int32((target-cfg.TRAIN.MOV)/cfg.TRAIN.STRIDE)))
            ww=int(index[2]-index[0])
            hh=int(index[3]-index[1])
            labelcls2=np.zeros((1,size,size))-2
            labelcls2[0,index[1]:index[3]+1,index[0]:index[2]+1]=-1
            labelcls2[0,index[1]+hh//4:index[3]+1-hh//4,index[0]+ww//4:index[2]+1-ww//4]=1  
            overlap[i] = IoU([x1, y1, x2, y2], target)
            pos1 = np.where((overlap[i] > 0.80)) 
            neg1 = np.where((overlap[i] <= 0.5))
            pos1, pos_num1 = self.select(pos1, cfg.TRAIN.POS_NUM)
            neg1, neg_num1 = self.select(neg1, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)
            cls[i][pos1] = 1
            cls[i][neg1] = 0
            pos = np.where((overlap[i] > 0.72))
            neg = np.where((overlap[i] <= 0.45)) 
            pos, pos_num = self.select(pos, cfg.TRAIN.POS_NUM)
            neg, neg_num = self.select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)
            if anchor[:,2].min()>0 and anchor[:,3].min()>0:    
                delta[i][0] = (tcx - cx) / (w+1e-6)
                delta[i][1] = (tcy - cy) / (h+1e-6)     
                delta[i][2] = np.log(tw / (w+1e-6) + 1e-6)
                delta[i][3] = np.log(th / (h+1e-6) + 1e-6)
                delta_weight[i][pos] = 1. / (pos_num + 1e-6)
                delta_weight[i][neg] =0 

        cls=t.Tensor(cls).cuda()
        delta_weight=t.Tensor(delta_weight).cuda()
        delta=t.Tensor(delta).cuda()
        
        return cls, delta, delta_weight
