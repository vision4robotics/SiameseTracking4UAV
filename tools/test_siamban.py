# Parts of this code come from https://github.com/hqucv/siamban .
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from snot.core.config_ban import cfg
from snot.models.siamban_model import ModelBuilderBAN
from snot.trackers.tracker_builder_ban import build_tracker_ban
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain
from snot.datasets import DatasetFactory, datapath


torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='siamban tracking')
parser.add_argument('--dataset', default='', type=str,
                    help='datasets')
parser.add_argument('--datasetpath', default='', type=str,
                    help='the path of datasets')
parser.add_argument('--config', default='../experiments/SiamBAN/config.yaml', type=str,
                    help='config file')
parser.add_argument('--snapshot', default='../experiments/SiamBAN/model.pth', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--vis', default='', action='store_true',
                    help='whether visualzie result')
parser.add_argument('--trackername', default='SiamBAN', type=str,
                    help='name of tracker')
args = parser.parse_args()


def main():
    # load config
    cfg.merge_from_file(args.config)

    # create model
    model = ModelBuilderBAN()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker_ban(model)


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
            scores = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img)  
                    pred_bbox = outputs['bbox']
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])
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
