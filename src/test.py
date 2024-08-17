import numpy as np
import motmetrics as mm

#This script is used to evaluate the tracking results using the MOTChallenge metrics for a single sequence. 
def track_results():
    #/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/test/uav0000013_00000_v/gt/gt.txt
    #/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/test/001/gt/gt.txt
    # Load the ground truth data
   # gt = np.loadtxt("/home/abdulbhutta/Desktop/working-0809/MCMOT-master/dataset/hoot/annos/apple/001/gt.txt", delimiter=",")
    gt = np.loadtxt("/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/labels_with_ids/test/bottle/009/gt.txt", delimiter=",")
    gt_frame_ids = gt[:, 0].astype(int)
    gt_ids = gt[:, 1].astype(int)
    gt_boxes = gt[:, 2:6]

    #print("gt_ids: ", gt_ids)
    #print("gt_frame_ids: ", gt_frame_ids)

    # Load the results data
    #/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/results/MOT15_val_all_dla34/uav0000013_00000_v.txt
    #/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/results/MOT15_val_all_dla34/001.txt
    results = np.loadtxt("/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/images/results/MOT15_val_all_dla34/bottle/009.txt", delimiter=",")
    #results_apple = np.loadtxt("/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/results/MOT15_val_all_dla34/001.txt", delimiter=",")
    res_frame_ids = results[:, 0].astype(int)
    res_ids = results[:, 1].astype(int)
    res_boxes = results[:, 2:6]
    #print("res_ids: ", res_ids) 
    #print("res_frame_ids: ", res_frame_ids)

    # Create an accumulator
    #auto_id=True: Automatically assign unique object IDs to the ground truth and result data
    acc = mm.MOTAccumulator(auto_id=True)

    # Update the accumulator with the ground truth and results data
    for frame_id in np.unique(gt_frame_ids):
        gt_indices = np.where(gt_frame_ids == frame_id)
        res_indices = np.where(res_frame_ids == frame_id)

        gt_boxes_frame = gt_boxes[gt_indices]
        res_boxes_frame = res_boxes[res_indices]

        #print("gt_boxes_frame: ", gt_boxes_frame)
        #print("track_ids: ", res_boxes_frame)

        dists = mm.distances.iou_matrix(gt_boxes_frame, res_boxes_frame, max_iou=0.5)

        acc.update(
            gt_ids[gt_indices],
            res_ids[res_indices],
            dists
        )

    # Compute the metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'idp', 'idr', 'precision', 'recall', 'num_switches', 'track_ratios'] , name='acc')
    print(summary)

track_results()
