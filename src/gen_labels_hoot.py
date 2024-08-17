import cv2
import numpy as np
import os
import os.path as osp
import json 
import tqdm
import sys

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

#generate video from images
def gen_video(img_dir, output_name, output_video_path):

    if not osp.exists(output_video_path):
        os.makedirs(output_video_path)  

    img_dir = img_dir
    output_name = output_name
    output_video_path = '{}{}.mp4'.format(output_video_path, output_name)

    #000000.png
    cmd_str = 'ffmpeg -f image2 -i {}/%06d.png -b 5000k -c:v mpeg4 {}' .format(img_dir, output_video_path)
    os.system(cmd_str)
    print('Video generated at: {}'.format(output_video_path))


#Generate labels for the Hoot dataset
def gen_labels_hoot():
    video_root =  "/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/images/train"
    ground_truth_root = "/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/labels_with_ids/train"

    if not osp.exists(ground_truth_root):
        os.makedirs(ground_truth_root)

    #Folder: /home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/images/train
    object_classes = sorted(os.listdir(video_root))
    
    #Map each object classes to a number 
    object_classes = {object_class: i for i, object_class in enumerate(object_classes)}
    print('Object classes: {}'.format(object_classes))

    #For each object class in the folder
    for object_class in object_classes:
        object_class_path = osp.join(video_root, object_class)
        print('Object class path: {}'.format(object_class_path))

        #Get all the sequences in the class
        sequences = sorted(os.listdir(object_class_path))
        print('Sequences: {}'.format(sequences))

        #For each sequence in the class
        for seq in sequences:
            video_info = open(osp.join(object_class_path, seq, 'meta.info')).read()
            width = video_info[video_info.find('"width": ') + 9:video_info.find('",\n   "height"')]
            height = video_info[video_info.find('"height": ') + 10:video_info.find('",\n   "width_orig"')]

            #strip white spaces
            width = width.strip('"').strip()
            height = height.strip('"').strip()
            
            #convert to int
            width = int(width)
            height = int(height)

            #read the gt file
            gt_file = osp.join(object_class_path, seq, 'anno.json')
            gt = json.loads(open(gt_file).read())

            #print the object class, sequence, gt file, width and height
            print('-' * os.get_terminal_size().columns)
            print('Object class: {}'.format(object_class))
            print('Sequence: {}'.format(seq))
            print('GT file: {}'.format(gt_file))
            print('Width: {}, Height: {}'.format(width, height))
            mkdirs(osp.join(ground_truth_root, object_class, seq))

            #For each frame in the sequence
            for frames in tqdm.tqdm(gt["frames"]):
                    #if the first frame then 
                    if frames["frame_id"] == 0:
                        tqdm.tqdm.write("Processing sequence: {}".format(seq))
                        #print("Processing sequence: {}".format(seq))

                    frame_id = frames["frame_id"]
                    aa_bb = frames["aa_bb"]
                    #print ('Frame ID: {}, AA_BB: {}'.format(frame_id, aa_bb))

                    # Skip the frame if there are no bounding boxes
                    if len(aa_bb) == 0:
                        continue

                    # Extract x and y coordinates separately
                    x_coords = [point[0] for point in aa_bb]
                    y_coords = [point[1] for point in aa_bb]

                    # Calculate min and max for x and y coordinates
                    min_x = min(x_coords)
                    max_x = max(x_coords)
                    min_y = min(y_coords)
                    max_y = max(y_coords)

                    # Calculate the center of the bounding box
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2

                    # Calculate the width and height of the bounding box
                    width_bb = max_x - min_x
                    height_bb = max_y - min_y

                    # Normalize the center, width, and height
                    norm_center_x = center_x / width
                    norm_center_y = center_y / height
                    norm_width_bb = width_bb / width
                    norm_height_bb = height_bb / height

                    top_left_x = aa_bb[0][0]
                    top_left_y = aa_bb[0][1]

                    track_id = 1 # only one object in the frame

                    #show the frame with the bounding box 
                    img_path = osp.join(video_root, object_class, seq, '{:06d}.png'.format(frame_id))
                    img = cv2.imread(img_path)
                    cv2.rectangle(img, (int(aa_bb[0][0]), int(aa_bb[0][1])), (int(aa_bb[2][0]), int(aa_bb[2][1])), (0, 255, 0), 2)
                    cv2.imshow('Frame', img)

                    #plot the center of the bounding box
                    cv2.circle(img, (int(center_x), int(center_y)), 5, (0, 0, 255), -1) #cv2.cirlce(img, center, radius, color, thickness)
                    cv2.imshow('Frame', img)
                    
                    #plot only the top left corner of the bounding box with a red circle and the track id as text on the image
                    cv2.circle(img, (int(top_left_x), int(top_left_y)), 5, (255, 0, 0), -1)

                    #top left corner of the bounding box with track id
                    cv2.putText(img, str("Track ID: {}".format(track_id)), (int(top_left_x), int(top_left_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 

                    #put the class of the object on the image at center top of the bounding box
                    cv2.putText(img, object_class, (int(center_x), int(center_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    cv2.imshow('Frame', img)
                    cv2.waitKey(1)

                    #if the width and height of the bounding box is less than 25 pixels, skip the frame
                    if width_bb < 15 or height_bb < 15:
                        #add the skipped frame id to the skipped_frames.txt file in the data folder
                        with open("/home/abdulbhutta/Desktop/MCMOT-master/data/skipped_frames.txt", 'a') as f:
                            f.write("Object class: {}, Sequence: {}, Skipped frame: {}\n".format(object_class, seq, frame_id))
                        continue

                    #if apple seq 003 then save the image with the bounding box and the center of the bounding box to the folder: /home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/saved_bb/
                    if object_class == 'apple' and seq == '003':
                        saved_bb_path = "/home/abdulbhutta/Desktop/MCMOT-master/data/saved_bb"
                        mkdirs(saved_bb_path)
                        saved_bb_path = osp.join(saved_bb_path, '{:06d}.png'.format(frame_id))
                        cv2.imwrite(saved_bb_path, img)
                        #print('Image saved at: {}'.format(saved_bb_path))

                    #Convert to format of the line is: "class track_id x_center/img_width y_center/img_height w/img_width h/img_height". (Normalized format)
                    frame_path = osp.join(ground_truth_root, object_class, seq, '{:06d}.txt'.format(frame_id))
                    with open(frame_path, 'w') as f:
                        f.write('{} 1 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(object_classes[object_class], norm_center_x, norm_center_y, norm_width_bb, norm_height_bb))
                        #print('Frame path: {}'.format(frame_path))
                    
            print("Finished processing sequence: {}".format(seq))

#Generate path to images in the dataset
def gen_path_to_images():
    #Path to the dataset
    dataset_path = "/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/labels_with_ids/train"
    relative_path = "hoot/images/train"
        
    object_classes = sorted(os.listdir(dataset_path))
    object_classes = {object_class: i for i, object_class in enumerate(object_classes)}
    print('Object classes: {}'.format(object_classes))

    #Get all the folders in the dataset
    for folder in sorted(os.listdir(dataset_path)):
        #print("Folder: ", folder)
        folder_path = os.path.join(dataset_path, folder)
        #print("Folder path: ", folder_path)

        #Get all the subfolders in the folder
        for subfolder in sorted(os.listdir(folder_path)):
            print("Subfolder: ", subfolder)
            subfolder_path = os.path.join(folder_path, subfolder)
            print("Subfolder path: ", subfolder_path)

            #Get all the images in the subfolder
            for img in sorted(os.listdir(subfolder_path)):
                img = os.path.join(img)

                #Pnly get the images with the .png extension
                if img.endswith('.txt'):
                    #print("Image path: ", img)

                    #Remove txt and add .png
                    img = img.replace('.txt', '.png')

                    save_path = osp.join(folder_path, subfolder, img)
                    #Write the path to the text file and if not exist then create it in location /home/abdulbhutta/Desktop/MCMOT-master/data/train.txt
                    with open("/home/abdulbhutta/Desktop/MCMOT-master/data/full_train.txt", 'a') as f:
                        f.write(relative_path + '/' + folder + '/' + subfolder + '/' + img + '\n')

#Generate path with the first image only in the sequence (Testing purposes)
def gen_path_to_images_first():
    f = open("/home/abdulbhutta/Desktop/MCMOT-master/data/train.txt", 'w')
    dataset_path = "/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/labels_with_ids/train"
    relative_path = "hoot/images/train"

    object_classes = sorted(os.listdir(dataset_path))
    object_classes = {object_class: i for i, object_class in enumerate(object_classes)}
    print('Object classes: {}'.format(object_classes))

    #Get all the folders in the dataset
    for folder in sorted(os.listdir(dataset_path)):
        #print("Folder: ", folder)
        folder_path = os.path.join(dataset_path, folder)
        #print("Folder path: ", folder_path)

        #Get all the subfolders in the folder
        subfolders_list = sorted(os.listdir(folder_path))
        print("Subfolders list: ", subfolders_list)

        for subfolder in sorted(os.listdir(folder_path)):
            if subfolder == subfolders_list[0]:
                print("Current subfolder: ", subfolder)
                
                #print("Subfolder: ", subfolder)
                subfolder_path = os.path.join(folder_path, subfolder)
                #print("Subfolder path: ", subfolder_path)

                #Get all the images in the subfolder
                for img in sorted(os.listdir(subfolder_path)):
                    img = os.path.join(img)

                    #Only get the images with the .png extension
                    if img.endswith('.txt'):
                        #print("Image path: ", img)

                        #remove txt and add .png
                        img = img.replace('.txt', '.png')
                    
                        save_path = osp.join(folder_path, subfolder, img)
                        #write the path to the text file and if not exist then create it in location /home/abdulbhutta/Desktop/MCMOT-master/data/train.txt
                        with open("/home/abdulbhutta/Desktop/MCMOT-master/data/first_train.txt", 'a') as f:
                            f.write(relative_path + '/' + folder + '/' + subfolder + '/' + img + '\n')

if __name__ == "__main__":
    #Arguments 
    #1 for gen_labels_hoot.py
    #2 for gen_path_to_images.py
    #3 for gen_path_to_images_first.py
    #4 for gen_video.py
    #5 for track_results.py

    if sys.argv[1] == "1":
        gen_labels_hoot()
        #gen_labels_hoot_one_seq()
    elif sys.argv[1] == "2":
        gen_path_to_images()
    elif sys.argv[1] == "3":
        gen_path_to_images_first()  
    else:
        print("Invalid argument")
