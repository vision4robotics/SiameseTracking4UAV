# Copyright (c) 2021 Multimedia Research.
# Parts of this code come from https://github.com/researchmm/LightTrack .
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from easydict import EasyDict as edict
from snot.models import lighttrack_model as models
from snot.trackers.lighttrack_tracker import Lighttrack
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain
from snot.datasets import DatasetFactory, datapath


torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='lighttrack tracking')
parser.add_argument('--dataset', default='',type=str,
                    help='datasets')
parser.add_argument('--datasetpath', default='', type=str,
                    help='the path of datasets')
parser.add_argument('--arch', dest='arch', default='LightTrackM_Subnet',
                    help='backbone architecture')
parser.add_argument('--snapshot', default='../experiments/LightTrack/model.pth', type=str,
                    help='pretrained model')
parser.add_argument('--path_name', default='back_04502514044521042540+cls_211000022+reg_100000111_ops_32', type=str,
                    help='path name')
parser.add_argument('--stride', default=16, type=int,
                    help='network stride')
parser.add_argument('--even', default=0, type=int,
                    help='even')
parser.add_argument('--epoch_test', default=False, type=bool, 
                    help='multi-gpu epoch test flag')
parser.add_argument('--video', default='', type=str,    
                    help='eval one special video')
parser.add_argument('--vis', default='', action='store_true',
                    help='whether visualzie result')
parser.add_argument('--trackername', default='LightTrack', type=str,
                    help='name of tracker')
args = parser.parse_args()


def main():
    # prepare model
    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.epoch_test = args.epoch_test
    info.stride = args.stride
    net = models.__dict__[args.arch](args.path_name, stride=info.stride)
    net = load_pretrain(net, args.snapshot)
    net.eval()
    net = net.cuda()

    # prepare tracker
    tracker = Lighttrack(info, even=args.even)

    # set hyper parameters
    hp = {'LightTrackM_Subnet':{'penalty_k': 0.007, 'lr': 0.616, 'window_influence': 0.225, 'small_sz': 256, 'big_sz': 288, 'ratio': 1}}


    for dataset_name in args.dataset.split(','):
        # create dataset
        try:
            dataset_root = args.datasetpath + datapath[dataset_name]
        except:
            print('?')
        dataset = DatasetFactory.create_dataset(name=dataset_name,
                                            dataset_root=dataset_root,
                                            load_img=False)
        model_name = args.trackername
        
        # OPE tracking
        IDX = 0
        TOC = 0
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    target_pos = np.array([cx, cy])
                    target_sz = np.array([w, h])
                    state = tracker.init(img, target_pos, target_sz, net, hp[args.arch])
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(pred_bbox)
                else: 
                    state = tracker.track(state, img)
                    target_pos=state['target_pos']
                    target_sz=state['target_sz']
                    pred_bbox=np.array([target_pos[0]-target_sz[0]/2, target_pos[1]-target_sz[1]/2, target_sz[0], target_sz[1]])
                    pred_bboxes.append(pred_bbox)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    try:
                        gt_bbox = list(map(int, gt_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    except:
                        pass
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results 
            model_path = os.path.join('results', dataset_name, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(v_idx+1, video.name, toc, idx / toc))
            IDX += idx
            TOC += toc
        print('Total Time: {:5.1f}s Average Speed: {:3.1f}fps'.format(TOC, IDX / TOC))
        fps_path = os.path.join('results', dataset_name, '{}.txt'.format(model_name))
        with open(fps_path, 'w') as f:
            f.write('Time:{:5.1f},Speed:{:3.1f}'.format(TOC, IDX / TOC))


if __name__ == '__main__':
    main()
