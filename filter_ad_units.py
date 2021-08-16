import cv2
import json
import os,sys
import numpy as np
from shapely.geometry import *
import shapely
import ctypes
from ctypes import *

cudasift = cdll.LoadLibrary(os.getcwd()+ "/libcudasiftlib.so")

class RetStruct(Structure):
    _fields_ =[('numPts', c_int),
               ('x_pos', POINTER(c_float)),
               ('y_pos', POINTER(c_float)),
               ('m_x_pos', POINTER(c_float)),
               ('m_y_pos', POINTER(c_float)),
               ('match_error', POINTER(c_float))]


def crop_image(image, points, extend_ratio=0.1):
    xmin, xmax, ymin, ymax = points[0][0], points[0][0], points[0][1], points[0][1]
    for k in range(1, 4):
        xmin = min(xmin, points[k][0])
        ymin = min(ymin, points[k][1])
        xmax = max(xmax, points[k][0])
        ymax = max(ymax, points[k][1])

    extend_x = (xmax - xmin) * extend_ratio
    extend_y = (ymax - ymin) * extend_ratio

    h, w = image.shape[:2]
    bbox = [max(0, xmin - extend_x), max(0, ymin - extend_y), min(w, xmax + extend_x), min(h, ymax + extend_y)]

    t_croped_img = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    croped_img = np.zeros(t_croped_img.shape,dtype = np.uint8)
    for i in range(croped_img.shape[0]):
        for j in range(croped_img.shape[1]):
            croped_img[i][j] = t_croped_img[i][j]
    return croped_img

def sift_match(img1_des, img2_des,img1_key,img2_key,corner1,corner2):
    STANDARD_AD_UNIT_AREA = 200 * 200

    if img1_des is None or img2_des is None:
        return 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(img1_des, img2_des, k=2)

    if len(matches) == 0 or len(matches[0]) != 2:
        return  0

    good_match = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_match.append(m)

    kp1 = []
    kp2 = []
    for match in good_match:
        kp1.append([img1_key[match.queryIdx].pt[0],img1_key[match.queryIdx].pt[1]])
        kp2.append([img2_key[match.trainIdx].pt[0],img2_key[match.trainIdx].pt[1]])

    kp1 = np.array(find_kp_position(kp1,corner1),np.int)
    kp2 = np.array(find_kp_position(kp2,corner2),np.int)
    match_score = compute_if_same_corner(corner1,corner2,kp1,kp2)
    return match_score

def find_kp_position(kps,corner,extend_ratio = 0.1):
    xmin, xmax, ymin, ymax = corner[0][0], corner[0][0], corner[0][1], corner[0][1]
    for k in range(1, 4):
        xmin = min(xmin, corner[k][0])
        ymin = min(ymin, corner[k][1])
        xmax = max(xmax, corner[k][0])
        ymax = max(ymax, corner[k][1])
    extend_x = (xmax - xmin) * extend_ratio
    extend_y = (ymax - ymin) * extend_ratio
    corner_posx = max(0, xmin-extend_x)
    corner_posy = max(0, ymin-extend_y)
    for kp in kps:
        kp[0]+=corner_posx
        kp[1]+=corner_posy 
    return kps


def maskflownet_match(readimage1,readimage2,predictor,corner1,corner2):
    def find_xy_based_corner(corner):
        x_max,y_max = np.array(corner).max(axis=0)
        x_min,y_min = np.array(corner).min(axis=0)
        return x_max,x_min,y_max,y_min

    flow = predictor.predict(readimage1,readimage2)
    x1_max,x1_min,y1_max,y1_min = find_xy_based_corner(corner1)
    match_points1 = []
    match_points2 = []
    for x1 in range(x1_min,x1_max):
        for y1 in range(y1_min,y1_max):
            match_points1.append([x1,y1])
            match_points2.append([(int)(x1+flow[y1][x1][1]),(int)(y1+flow[y1][x1][0])])
    return compute_if_same_corner(corner1,corner2,match_points1,match_points2)

def detect_corner_sift(corner1,match_point1,match_point2):
    def find_new_box_based_perspectivetransformer(corner,M):
        corner_t = np.float32(corner)
        corner_t = corner_t.reshape(-1, 1, 2)
        corner2_t = cv2.perspectiveTransform(corner_t, M)
        ret_corner = corner2_t.reshape(4,2)
        ret_corner = np.array(ret_corner,np.int)
        return ret_corner
    #print(match_point1,match_point2)
    if match_point1 is None or len(match_point1)<10:
        return None
    M, mask = cv2.findHomography(np.asarray(match_point1), np.asarray(match_point2), cv2.RANSAC, 5.0)
    if M is None:
        return None
    detect_corner_box = find_new_box_based_perspectivetransformer(corner1,M)
    return detect_corner_box

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


def gpu_sift(img1,img2,corner1):

    img1d = cv2.imread(img1,0 )
    img2d = cv2.imread(img2,0 )
    img1d = crop_image(img1d, corner1)
    #img1d = np.ascontiguousarray(img1, dtype=np.uint8)
    np.array(img1).flatten()
    #print(img1d.shape)
    #img2d = np.ascontiguousarray(img2, dtype=np.uint8)
    np.array(img2).flatten()
    #print(img2d.shape)
    frame_data1 = np.asarray(img1d, dtype=np.uint8)
    frame_data1 = frame_data1.ctypes.data_as(ctypes.c_char_p)
    frame_data2 = np.asarray(img2d, dtype=np.uint8)
    frame_data2 = frame_data2.ctypes.data_as(ctypes.c_char_p)
    h1,w1=img1d.shape[0],img1d.shape[1] 
    h2,w2=img2d.shape[0],img2d.shape[1] 
    #print(h1,w1,h2,w2)
    cudasift.sift_process.restype = POINTER(RetStruct)
    t = cudasift.sift_process(h1,w1,frame_data1,h2,w2,frame_data2,1)
    return t

def gpu_sift_detect(img1,img2,corner1):
    keypoints = gpu_sift(img1,img2,corner1)
    kp1 = []
    kp2 = []
    
    for i in range(keypoints.contents.numPts):
        kp1.append([keypoints.contents.x_pos[i],keypoints.contents.y_pos[i]])
        kp2.append([keypoints.contents.m_x_pos[i],keypoints.contents.m_y_pos[i]])

    kp1 = np.array(find_kp_position(kp1,corner1),np.int)
    detect_corner = detect_corner_sift(corner1,kp1,kp2)
    return detect_corner


def match_images(raw_images,corners,images, frame_idxes, frame_lookback, thre=0.6):

    def get_merge_set_parent(p, x):
        if p[x] == x:
            return x
        p[x] = get_merge_set_parent(p, p[x])
        return p[x]

    def merge_set_union(p, x, y):
        p_x = get_merge_set_parent(p, x)
        p_y = get_merge_set_parent(p, y)
        p[p_x] = p[p_y]

    N = len(images)
    
    sift = cv2.SIFT_create()
    descriptors_and_keypoints = np.array([sift.detectAndCompute(img, None) for img in images])
    descriptors = descriptors_and_keypoints[:,1]
    keypoints = descriptors_and_keypoints[:,0]
    
    p = [k for k in range(N)]
    f = open('tmp2.txt','w')
    for i in range(1, N):
        idx_i = frame_idxes[i]
        des_i = descriptors[i]
        key_i = keypoints[i]
        max_score = 0
        matched_idx = -1
        for j in range(i - 1, -1, -1):
            idx_j = frame_idxes[j]
            if idx_j + frame_lookback < idx_i:
                break
            print(frame_idxes[j],frame_idxes[i])
            key_points = gpu_sift(images[j],raw_images[i])
            match_score_gpu = gpu_sift_match(key_points,corners[i],corners[j])
            des_j = descriptors[j]
            key_j = keypoints[j]
            match_score = sift_match(des_i, des_j,key_i,key_j,corners[i],corners[j])
            if(abs(match_score_gpu-match_score)>0.3 and match_score>thre):
                f.write(str(abs(match_score_gpu-match_score)))
                f.write('   '+str(frame_idxes[i]) + ' ' +str(frame_idxes[j]))
                f.write(r'\n')
            print('si',i,j,frame_idxes[i],frame_idxes[j],match_score)
            print('si_g',i,j,frame_idxes[i],frame_idxes[j],match_score_gpu)
            if match_score > thre:
                max_score = match_score
                matched_idx = j
                break

        if max_score > thre:
            merge_set_union(p, i, matched_idx)
    f.close()
    match_results = {}

    for k in range(N):
        p_k = get_merge_set_parent(p, k)
        if p_k == k:
            match_results[k] = [k]
        else:
            match_results[p_k].append(k)
    
    return match_results

def filter_ad_unit_by_duration(cfg, dict_ad_units, ad_unit_duration_thre=2):
    dict_ad_units.sort(key=lambda x: x['frameid'])
    ad_unit_imgs = []
    frame_indexes = []
    ad_unit_imgs_corners = []
    ad_unit_raw_imgs = []
    frame_lookback = cfg.VIDEO_FPS * 5

    for ad_unit in dict_ad_units:
        frameid = (int)(ad_unit['frameid'])
        corners = ad_unit['corners']
        img = cv2.imread(os.path.join(cfg.DATA_FOLDER, cfg.FRAMES_FOLDER, 'ad_unit_%d.jpg' % (frameid)),0)
        croped_img = crop_image(img, corners)
        ad_unit_imgs.append(croped_img)
        frame_indexes.append(frameid)
        ad_unit_imgs_corners.append(corners)
        ad_unit_raw_imgs.append(img)
    
    ad_unit_match_result = match_images(ad_unit_raw_imgs,ad_unit_imgs_corners,ad_unit_imgs, frame_indexes, frame_lookback)
    print(ad_unit_match_result)
    frame_interval_thre = ad_unit_duration_thre * cfg.VIDEO_FPS
    filtered_ad_units = []
    for k in ad_unit_match_result:
        matched_ad_unit_indexes = ad_unit_match_result[k]
        frame_interval = frame_indexes[matched_ad_unit_indexes[-1]] - frame_indexes[matched_ad_unit_indexes[0]]
        if frame_interval < frame_interval_thre:
            continue
        #matched_ad_unit_indexes.sort(key=lambda x: dict_ad_units[x]['score'])
        filtered_ad_units.append(dict_ad_units[matched_ad_unit_indexes[-1]])
    
    return filtered_ad_units

def filter_ad_unit_by_min_score(dict_ad_units, min_ad_unit_score=0.3):
    filtered_ad_units = []
    for ad_unit in dict_ad_units:
        if ad_unit['score'] > min_ad_unit_score:
            filtered_ad_units.append(ad_unit)
    return filtered_ad_units

def filter_ad_unit_by_ranking_and_interval(cfg, dict_ad_units, second_per_ad=120, time_interval=60):
    keep_ad_num = int(cfg.VIDEO_N_FRAMES / cfg.VIDEO_FPS / second_per_ad)
    if keep_ad_num < 1:
        keep_ad_num = 1
    frame_id_interval = cfg.VIDEO_FPS * time_interval
    dict_ad_units.sort(key=lambda s:s['score'], reverse=True)
    keeped_ad_units = []
    for k in range(len(dict_ad_units)):
        if len(keeped_ad_units) >= keep_ad_num:
            break
        keep = True
        for i in range(len(keeped_ad_units)):
            if abs(dict_ad_units[k]['frameid'] - keeped_ad_units[i]['frameid']) < frame_id_interval:
                keep = False
                break
        if keep:
            keeped_ad_units.append(dict_ad_units[k])

    keeped_ad_units.sort(key=lambda s:s['frameid'])
    return keeped_ad_units

def filter_ad_units_by_score_and_duration(cfg, dict_ad_units):
    filtered_ad_units = filter_ad_unit_by_min_score(dict_ad_units)
    filtered_ad_units = filter_ad_unit_by_duration(cfg, filtered_ad_units)
    filtered_ad_units.sort(key=lambda s:s['frameid'])
    return filtered_ad_units

def filter_ad_units_by_pre_tracking_result(cfg, dict_ad_units):
    pre_tracking_result_json_path = os.path.join(cfg.DATA_FOLDER, cfg.AD_UNITS_TRACKING.MODIFIED_AD_UNITS_JSONS, cfg.AD_UNITS_TRACKING.TRACKING_RESULT_FOR_DETECTION)
    with open(pre_tracking_result_json_path) as f:
        j = json.load(f)
    if j['n_instances'] == 0:
        return []
    instance_valid_dict = j['instance_valid']
    filtered_ad_units = []
    for idx in instance_valid_dict:
        if instance_valid_dict[idx] == True:
            filtered_ad_units.append(dict_ad_units[int(idx)])
    filtered_ad_units.sort(key=lambda s:s['frameid'])
    return filtered_ad_units
