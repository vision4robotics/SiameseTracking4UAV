import os
import sys
import argparse
sys.path.append("./")

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from snot.datasets import *
from snot.utils.evaluation import OPEBenchmark
from snot.utils.visualization import draw_success_precision


parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
parser.add_argument('--datasetpath', default='',type=str, help='dataset root directory')
parser.add_argument('--dataset', default='',type=str, help='dataset name')
parser.add_argument('--tracker_result_dir',default='', type=str, help='tracker result root')
parser.add_argument('--tracker_path', default='.', type=str)
parser.add_argument('--tracker_prefix',default='', type=str)
parser.add_argument('--vis', default='',dest='vis', action='store_true')
parser.add_argument('--show_video_level', default='',dest='show_video_level', action='store_true')
parser.add_argument('--num', default=1, type=int, help='number of processes to eval')
args = parser.parse_args()


def main():
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                  args.dataset,
                                  args.tracker_prefix))
    trackers = [x.split('/')[-1] for x in trackers]

    root = args.dataset_dir + args.dataset

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if 'UAV10' in args.dataset:
        dataset = UAV10Dataset(args.dataset, root)
    elif 'UAV20' in args.dataset:
        dataset = UAV20Dataset(args.dataset, root)
    elif 'DTB70' in args.dataset:
        dataset = DTBDataset(args.dataset, root)
    elif 'UAVDT' in args.dataset:
        dataset = UAVDTDataset(args.dataset, root)
    elif 'VISDRONED' in args.dataset:
        dataset = VISDRONEDDataset(args.dataset, root)
    elif 'UAVTrack112' in args.dataset:
        dataset = V4RDataset(args.dataset, root)
    elif 'UAVDark135' in args.dataset:
        dataset = UAVDARKDataset(args.dataset, root)
    elif 'DarkTrack2021' in args.dataset:
        dataset = DARKTRACKDataset(args.dataset, root)
    elif 'NAT2021' in args.dataset:
        dataset = DARKTRACKDataset(args.dataset, root)


    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    success_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
            trackers), desc='eval success', total=len(trackers), ncols=18):
            success_ret.update(ret)
    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
            trackers), desc='eval precision', total=len(trackers), ncols=18):
            precision_ret.update(ret)
    benchmark.show_result(success_ret, precision_ret,
            show_video_level=args.show_video_level)
    if args.vis:
        for attr, videos in dataset.attr.items():
            draw_success_precision(success_ret,
                        name=dataset.name,
                        videos=videos,
                        attr=attr,
                        precision_ret=precision_ret)

if __name__ == '__main__':
    main()