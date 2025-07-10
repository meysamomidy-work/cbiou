import os
import pickle
import numpy as np
from cbiou import CBIOUTracker
from track import *
from evaluate import evaluate
from utils import count_time

@count_time
def run():
    seqs = ['MOT17-04-FRCNN']

    os.makedirs('outputs/cbiou', exist_ok=True)

    with open(f"../../.Datasets/MOT17/train/mot17_val_0.80.pickle", 'rb') as f:
        detections_file = pickle.load(f)

    seqmap = open('./trackeval/seqmap/mot17/custom.txt', 'w')
    seqmap.write('name\n')
    for seq in seqs:
        
        seqmap.write(f'{seq}\n')
        file = open(f'outputs/cbiou/{seq}.txt', 'w')
        detections = detections_file[seq]
        gt_dets_file = np.loadtxt(f'../../.Datasets/MOT17/train/{seq}/gt/gt.txt', delimiter=',')
        gt_dets_file = gt_dets_file[gt_dets_file[:, 6] == 1]
        gt_dets_file = gt_dets_file[gt_dets_file[:, 7] == 1]

        cbiou = CBIOUTracker({
            'biou_threshold1': 0.7, 
            'biou_threshold2': 0.7, 
            'min_frames_to_predict': 3,
            'buffer_scale1': 0.3, 
            'buffer_scale2': 0.5, 
            'detection_threshold':0.2, 
            'max_age': 60})

        for i,frame_number in enumerate(np.unique(gt_dets_file[:,0])):
            gt_dets, dets = gt_dets_file[gt_dets_file[:,0] == frame_number][:, 1:6], detections[int(frame_number)][:, :5]
            cbiou.update(dets)
            for track in Track.INSTANCES:
                if track.state in [STATE_TRACKING, STATE_NEW]:
                    file.write(f'{track.mot_format}\n')
        file.close()
    seqmap.close()


if __name__ == '__main__':
    print('tracking...')
    run()
    print('evaluating...')
    evaluate()