# Multi-Class FairMOT-HOOT

## Abstract
This repo implements a multi-class FairMOT model and trains it using the Heavy Occlusion for Object Tracking (HOOT) dataset. The model is trained on a subset of the HOOT dataset containing 20 classes of objects and tested on 12 objects with 16 videos. It is based on the FairMOT and MCMOT repositories and modified for the HOOT dataset. To ensure the same results, it's crucial to follow the steps below or view step-by-step instructions in the [Jupyter Notebook](https://github.com/abdulbhutta/Multi-Class-FairMOT-HOOT/blob/main/AbdulBhutta_FairMOT_HOOT.ipynb).
 
<img src="https://github.com/abdulbhutta/Multi-Class-FairMOT-HOOT/blob/main/assets/track_apple.gif" alt="Description" width="49%" height="50%"/> <img src="https://github.com/abdulbhutta/Multi-Class-FairMOT-HOOT/blob/main/assets/track_carrot.gif" alt="Description" width="49%" height="50%"/> 
<img src="https://github.com/abdulbhutta/Multi-Class-FairMOT-HOOT/blob/main/assets/track_cat.gif" alt="Description" width="49%" height="50%"/> <img src="https://github.com/abdulbhutta/Multi-Class-FairMOT-HOOT/blob/main/assets/track_deer.gif" alt="Description" width="49%" height="50%"/> 

## Data Preparation

* **HOOT Dataset**
  
After downloading and extracting the dataset, place them in the follwing structure

```
hoot
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
   |         └——————train(empty)
   |         └——————test(empty)
```

* Follow instructions on setup and installation using FairMOT: [GitHub](https://github.com/ifzhang/FairMOT/tree/master)

## Model Training

Generate labels for the HOOT training dataset
```
cd src && python gen_labels_hoot.py 1
```
Generate path to image for training 
```
cd src && python gen_labels_hoot.py 2
```
Configure opts.py in src/lib/and update all the roots for dataset, models, and etc
```
--load_model
--data-dir
--data_cfg
--reid_cls_ids
```
Run the train.py script
```
cd src && python train.py 
```

## Model Testing

Update paths in gen_labels_hoot_test.py
```
video_root =  "/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/images/test"
ground_truth_root = "/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/labels_with_ids/test"
gt_path_root = '/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/images/test'
```
Generate ground truth annotations 
```
cd src && python gen_labels_hoot_test.py
```

Copy the ground truth labels to the current test dataset

Run the track.py script 
```
cd src && python track.py --test_hoot True
```
Run test.py to get evaluation for one image
```
cd src && python test.py
```

## Links
HOOT Dataset: [GitHub](https://github.com/gzdshn/hoot-toolkit)

Hoot Training/Testing Notebook: [GitHub](https://github.com/abdulbhutta/Multi-Class-FairMOT-HOOT/blob/main/AbdulBhutta_FairMOT_HOOT.ipynb)

Report: [GitHub](https://github.com/abdulbhutta/Multi-Class-FairMOT-HOOT/blob/main/AbdulBhutta_FinalReport.pdf)

Trained Model (Epoch 30): [Google Drive](https://drive.google.com/file/d/1M2xbOYVY7BTlAyB7T1WPBWWcjw5hzTVT/view?usp=sharing)

FairMOT: [GitHub](https://github.com/ifzhang/FairMOT/tree/master)

MCMOT: [GitHub](https://github.com/CaptainEven/MCMOT/tree/master)
