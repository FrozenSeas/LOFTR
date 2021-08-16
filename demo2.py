import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from shapely.geometry import *
import shapely
from src.loftr import LoFTR, default_cfg
from utils import *
from filter_ad_units import *
import datetime
# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.

def Detect_process(img0_pth,img1_pth):
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load("weights/indoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()
    # Load example images
    t_img0 = cv2.imread(img0_pth)
    t_img1 = cv2.imread(img1_pth)
    img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, (640, 480))
    img1_raw = cv2.resize(img1_raw, (640, 480))

    import datetime
    starttime = datetime.datetime.now()

    img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
    img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
    batch = {'image0': img0, 'image1': img1}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    endtime = datetime.datetime.now()

    color = cm.jet(mconf)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]
    """
    all_len = len(mkpts0)
    next_image1 = np.array(img0_raw)
    next_image2 = np.array(img1_raw)
    for i in range(all_len):
        next_image1[(int)(mkpts0[i][1])][(int)(mkpts0[i][0])] = 255
        next_image2[(int)(mkpts1[i][1])][(int)(mkpts1[i][0])] = 255
    concate_img = np.concatenate((next_image2,next_image1))
    cv2.imwrite(r'D:\LoTFR\Result\img.jpg',concate_img)
    """
    return mkpts0,mkpts1
"""
img0_pth = "assets/scannet_sample_images/scene0711_00_frame-001680.jpg"
img1_pth = "assets/scannet_sample_images/scene0711_00_frame-001995.jpg"
Detect_process(img0_pth,img1_pth)
"""
color = []
for i in range(2000):
    color.append((255,0,0))
    color.append((0,255,0))
    color.append((0,0,255))
def compute_if_same_corner(corner1,corner2,flow):
    low_x1 = min(min(min(corner1[0][0],corner1[3][0]),corner1[1][0]),corner1[2][0])
    low_y1 = min(min(min(corner1[0][1],corner1[3][1]),corner1[1][1]),corner1[2][1])
    high_x1 = max(max(max(corner1[0][0],corner1[3][0]),corner1[1][0]),corner1[2][0])
    high_y1 = max(max(max(corner1[0][1],corner1[3][1]),corner1[1][1]),corner1[2][1])
    low_x2 = min(min(min(corner2[0][0],corner2[3][0]),corner2[1][0]),corner2[2][0])
    low_y2 = min(min(min(corner2[0][1],corner2[3][1]),corner2[1][1]),corner2[2][1])
    high_x2 = max(max(max(corner2[0][0],corner2[3][0]),corner2[1][0]),corner2[2][0])
    high_y2 = max(max(max(corner2[0][1],corner2[3][1]),corner2[1][1]),corner2[2][1])
    U = 0
    I = 0
    for x1 in range(low_x1,high_x1):
        for y1 in range(low_y1,high_y1):
            y2 = (int)(flow[y1][x1][0]+y1)
            x2 = (int)(flow[y1][x1][1]+x1)
            if x2>=low_x2 and x2<=high_x2 and y2>=low_y2 and y2<=high_y2:
                I+=1
            else:
                U+=1
    U += (high_x2-low_x2)*(high_y2-low_y2)
    IOU = I/U
    if IOU>0.1:
        return True
    else:
        return False

def write_corners(image1,corner,color,seq):
    for i in range(len(corner)):
        start_point = (int(corner[i][0]),int(corner[i][1]))
        if i+1 < len(corner):
            end_point = (int(corner[i+1][0]),int(corner[i+1][1]))
        else:
            end_point = (int(corner[0][0]),int(corner[0][1]))
        thickness = 2
        image1 = cv2.line(image1, start_point, end_point, color[seq], thickness)
    return image1

def compute_corner_based_transformer(corner,kp1,kp2):
    low_x1 = min(min(min(corner[0][0],corner[3][0]),corner[1][0]),corner[2][0])
    low_y1 = min(min(min(corner[0][1],corner[3][1]),corner[1][1]),corner[2][1])
    high_x1 = max(max(max(corner[0][0],corner[3][0]),corner[1][0]),corner[2][0])
    high_y1 = max(max(max(corner[0][1],corner[3][1]),corner[1][1]),corner[2][1])
    kp_index = []
    for i in range(len(kp1)):
        if kp1[i][0]>=low_x1 and kp1[i][0]<=high_x1 and kp1[i][1]>=low_y1 and kp1[i][1]<=high_y1:
            kp_index.append(i)
    if len(kp_index) < 10:
        return None
    kp1_v = kp1[kp_index]
    kp2_v = kp2[kp_index]
    M, mask = cv2.findHomography(kp1_v, kp2_v, cv2.RANSAC, 5.0)
    corner_t = np.float32(corner)
    corner_t = corner_t.reshape(-1, 1, 2)
    corner2_t = cv2.perspectiveTransform(corner_t, M)
    corner2 = corner2_t.reshape(4,2)

    return corner2

def Detect_result_based_on_corner(images1,images2,corners,start,seqs,img_pts1,img_pts2):
    def find_xy_based_corner(corner):
        x_max,y_max = np.array(corner).max(axis=0)
        x_min,y_min = np.array(corner).min(axis=0)
        return x_max,x_min,y_max,y_min
    def compute_iou(corner1,corner2):
        a_polygon = Polygon(corner1)
        b_polygon = Polygon(corner2)
        if not a_polygon.intersects(b_polygon):
            iou = 0
        else:
            try:
                inter_area = a_polygon.intersection(b_polygon).area
                union_area = a_polygon.area + b_polygon.area - inter_area
                if union_area == 0:
                    return 1
                iou = float(inter_area) / union_area
            except shapely.geos.TopologicalError:
                iou = 0
        return iou 
    def Detect_result_based_on_corner_one_corner_pair_sift(image1,image2,corner1,corner2 = None):
        detect_corner2 = gpu_sift_detect(image1,image2,corner1)
        print(detect_corner2)
        if type(detect_corner2) != type(None):
            return detect_corner2
        return None
    def Detect_result_based_on_corner_one_corner_pair(image1,image2,kp1,kp2,corner1,corner2 = None):
        ratio_x = float(image1.shape[1]) / 640.0
        ratio_y = float(image1.shape[0]) / 480.0
        t_corner1 = np.array(corner1)
        #t_corner2 = np.array(corner2)
        for i in range(len(corner1)):
            t_corner1[i] = list(t_corner1[i])
            #t_corner2[i] = list(t_corner2[i])
            t_corner1[i][0] //= ratio_x
            t_corner1[i][1] //= ratio_y
            #t_corner2[i][0] //= ratio_x
            #t_corner2[i][1] //= ratio_y
        detect_corner2 = compute_corner_based_transformer(t_corner1,kp1,kp2)
        #image1 = write_corners(image1,corner1,[(255,0,0),(0,255,0),(0,0,255)],0)
        if type(detect_corner2) != type(None):
            for i in range(len(corner1)):
                detect_corner2[i][0] = detect_corner2[i][0] * ratio_x
                detect_corner2[i][1] = detect_corner2[i][1] * ratio_y
            return detect_corner2
        return None
    starttime = datetime.datetime.now()
    corner_list = []    #detail of corners
    corner_id_list = []  #class of corners
    corner_image_seq = []   #image_number of corners
    cur_corner_list = []
    cur_corner_id_list = []
    cur_corner_image_seq = []
    num_index = 0
    s_count = 0
    s_time = 0
    
    for seq in range(200000):
        if str(seq) in corners:
            for corner_id in corners[str(seq)]:
                corner_id_list.append(num_index)
                num_index+=1
                corner_list.append(corners[str(seq)][corner_id])
                corner_image_seq.append(str(seq*30+15))
    print(corner_list,corner_id_list,corner_image_seq)
    imw_time = 0
    for img_seq in range(start,seqs-1):
        starttime = datetime.datetime.now()
        if len(cur_corner_list)!=0 and str(cur_corner_image_seq[len(cur_corner_list)-1])==str(img_seq):
            for i in range(len(cur_corner_list)-1,-1,-1):
                if str(cur_corner_image_seq[i])!=str(img_seq):
                    break
                corner_infor = cur_corner_list[i]
                corner_img = img_seq
                for j in range(len(corner_id_list)):
                    if corner_id_list[j] == cur_corner_id_list[i]:
                        corner_infor = corner_list[j]
                        corner_img = corner_image_seq[j]
                        break
                detect_corner2 = Detect_result_based_on_corner_one_corner_pair_sift(img_pts1[(int)(corner_img)-start],img_pts2[img_seq-start],corner_infor)

                if detect_corner2 is not None and compare_img_hist(crop_image(images1[img_seq-start], cur_corner_list[i], extend_ratio=0.1),crop_image(images2[img_seq-start], detect_corner2, extend_ratio=0.1))<0.2:
                    cur_corner_list.append(detect_corner2)
                    cur_corner_id_list.append(cur_corner_id_list[i])
                    cur_corner_image_seq.append(img_seq+1)
                else:
                    kps1,kps2 = Detect_process(img_pts1[img_seq-start],img_pts2[img_seq-start])
                    detect_corner2 = Detect_result_based_on_corner_one_corner_pair(images1[img_seq-start],images2[img_seq-start],kps1,kps2,cur_corner_list[i])

                    if detect_corner2 is not None and compare_img_hist(crop_image(images1[img_seq-start], cur_corner_list[i], extend_ratio=0.1),crop_image(images2[img_seq-start], detect_corner2, extend_ratio=0.1))<0.2:
                        cur_corner_list.append(detect_corner2)
                        cur_corner_id_list.append(cur_corner_id_list[i])
                        cur_corner_image_seq.append(img_seq+1)
            if  str(img_seq+1) in corner_image_seq:
                t_corner_list = []    #detail of corners
                t_corner_id_list = []  #class of corners
                t_corner_image_seq = []   #image_number of corners
                for i in range(len(corner_image_seq)):
                    if corner_image_seq[i] == str(img_seq+1):
                        t_corner_list.append(corner_list[i])
                        t_corner_id_list.append(corner_id_list[i])
                        t_corner_image_seq.append(corner_image_seq[i])
                for i in range(len(cur_corner_list)-1,-1,-1):
                    if str(cur_corner_image_seq[i])!=str(img_seq+1):
                        break
                    for j in range(len(t_corner_image_seq)):
                        if compute_iou(t_corner_list[j],cur_corner_list[i])>0.5:
                            cur_corner_list[i] = t_corner_list[j]
                            t_corner_list.pop(j)
                            t_corner_id_list.pop(j)
                            t_corner_image_seq.pop(j)
                            break
                        if j == len(t_corner_image_seq)-1:
                            cur_corner_id_list.pop(i)
                            cur_corner_list.pop(i)
                            cur_corner_image_seq.pop(i)
                for i in range(len(t_corner_list)):
                    cur_corner_list.append(t_corner_list[i])
                    cur_corner_id_list.append(t_corner_id_list[i])
                    cur_corner_image_seq.append(t_corner_image_seq[i])
        else:
            if str(img_seq) in corner_image_seq:
                for i in range(len(corner_image_seq)):
                    if corner_image_seq[i] == str(img_seq):
                        cur_corner_list.append(corner_list[i])
                        cur_corner_id_list.append(corner_id_list[i])
                        cur_corner_image_seq.append(corner_image_seq[i])
                if len(cur_corner_list)!=0 and str(cur_corner_image_seq[len(cur_corner_list)-1])==str(img_seq):
                    for i in range(len(cur_corner_list)-1,-1,-1):
                        if str(cur_corner_image_seq[i])!=str(img_seq):
                            break
                        
                        detect_corner2 = Detect_result_based_on_corner_one_corner_pair_sift(img_pts1[img_seq-start],img_pts2[img_seq-start],cur_corner_list[i])
                                 
                        if detect_corner2 is not None and compare_img_hist(crop_image(images1[img_seq-start], cur_corner_list[i], extend_ratio=0.1),crop_image(images2[img_seq-start], detect_corner2, extend_ratio=0.1))<0.2:
                            cur_corner_list.append(detect_corner2)
                            cur_corner_id_list.append(cur_corner_id_list[i])
                            cur_corner_image_seq.append(img_seq+1)
                        else:
                            kps1,kps2 = Detect_process(img_pts1[img_seq-start],img_pts2[img_seq-start])

                            detect_corner2 = Detect_result_based_on_corner_one_corner_pair(images1[img_seq-start],images2[img_seq-start],kps1,kps2,cur_corner_list[i])
                            if detect_corner2 is not None and compare_img_hist(crop_image(images1[img_seq-start], cur_corner_list[i], extend_ratio=0.1),crop_image(images2[img_seq-start], detect_corner2, extend_ratio=0.1))<0.2:
                                cur_corner_list.append(detect_corner2)
                                cur_corner_id_list.append(cur_corner_id_list[i])
                                cur_corner_image_seq.append(img_seq+1)
                    
                    if str(img_seq+1) in corner_image_seq:
                        t_corner_list = []    #detail of corners
                        t_corner_id_list = []  #class of corners
                        t_corner_image_seq = []   #image_number of corners
                        for i in range(len(corner_image_seq)):
                            if corner_image_seq[i] == str(img_seq+1):
                                t_corner_list.append(corner_list[i])
                                t_corner_id_list.append(corner_id_list[i])
                                t_corner_image_seq.append(corner_image_seq[i])
                        for i in range(len(cur_corner_list)-1,-1,-1):
                            if str(cur_corner_image_seq[i])!=str(img_seq+1):
                                break
                            for j in range(len(t_corner_image_seq)):
                                if compute_iou(t_corner_list[j],cur_corner_list[i])>0.5:
                                    cur_corner_list[i] = t_corner_list[j]
                                    t_corner_list.pop(j)
                                    t_corner_id_list.pop(j)
                                    t_corner_image_seq.pop(j)
                                    break
                                if j == len(t_corner_image_seq)-1:
                                    cur_corner_id_list.pop(i)
                                    cur_corner_list.pop(i)
                                    cur_corner_image_seq.pop(i)
                        for i in range(len(t_corner_list)):
                            cur_corner_list.append(t_corner_list[i])
                            cur_corner_id_list.append(t_corner_id_list[i])
                            cur_corner_image_seq.append(t_corner_image_seq[i])
        
        endtime = datetime.datetime.now()
        cc = endtime - starttime
        imw_time += cc.seconds*1000000+cc.microseconds

        im = images1[img_seq-start]
        if len(cur_corner_list)!=0:
            for i in range(len(cur_corner_list)):
                if(str(cur_corner_image_seq[i])==str(img_seq)):
                    im = write_corners(images1[img_seq-start],cur_corner_list[i],color,cur_corner_id_list[i])
        cv2.imwrite('/root/LOFTR/Result_frames/' +'image'+ str(img_seq) + '.jpg',im)
    print(cur_corner_list,cur_corner_id_list,cur_corner_image_seq)
    print(imw_time)
