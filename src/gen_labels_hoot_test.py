import os
import os.path as osp
import json
import tqdm
import cv2


# Create a directory if it does not exist
def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

# Generate labels for the Hoot test dataset
def gen_labels_hoot():
    video_root =  "/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/images/test"
    ground_truth_root = "/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/labels_with_ids/test"

    if not osp.exists(ground_truth_root):
        os.makedirs(ground_truth_root)

    #get all the classes in the folder: /home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/images/train
    object_classes = sorted(os.listdir(video_root))
    
    #map each object classes to a number 
    object_classes = {object_class: i for i, object_class in enumerate(object_classes)}
    print('Object classes: {}'.format(object_classes))

    #only for the camel class
    #object_classes = {'camel': 14}

    for object_class in object_classes:
        object_class_path = osp.join(video_root, object_class)
        print('Object class path: {}'.format(object_class_path))

        #get all the sequences in the class
        sequences = sorted(os.listdir(object_class_path))
        print('Sequences: {}'.format(sequences))

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

                    gt_path_root = '/home/abdulbhutta/Desktop/MCMOT-master/dataset/hoot/images/test'
                    gt_path = osp.join(ground_truth_root, object_class, seq, 'gt.txt')
                    #MOT Format: <frame number>, <object id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence>, <x>, <y>, <z>
                    with open(gt_path, 'a+') as f:
                        f.write('{:d}, {:d}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, 1, -1, -1, -1 \n'.format(frame_id, track_id, top_left_x, top_left_y, width_bb, height_bb))  

            print("Finished processing sequence: {}".format(seq))

    print("Finished processing all sequences")
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    gen_labels_hoot()
