import cv2
import json
from pathlib import Path
import os
import numpy as np
"""
pre is the char before image number
dictionary_path is the path to all images
"""
def get_image_seqence_from_dir(dictionary_path,pre = None):
    seqs = []
    for p in Path(dictionary_path).iterdir():
        tmp = str(p).split('\\')
        tmp = tmp[len(tmp)-1]
        tmp = tmp.split(r'.')
        tmp = tmp[len(tmp)-2]
        if not pre == None:
            tmp = tmp.split(pre)
            tmp = tmp[len(tmp)-1]
        if not tmp in seqs:
            seqs.append(tmp)
    def Mysort(s):
        return (int)(s)
    seqs.sort(key = Mysort)
    return seqs

"""
pre is the prefix before image number
dictionary_path is the path to all images
"""
def generator_image_path_on_image_number(dictionary_path = '\\', image_number = 0, pre = None, file_type = '.jpg'):
    return os.path.join(dictionary_path,pre + str(image_number) + file_type)

def load_image_and_write_image_after_SIFT_Match(image1_path,image2_path,image1_number = '',image2_number = '',corners = None):
    ret_dict = {}
    readimage1 = cv2.imread(image1_path)
    readimage2 = cv2.imread(image2_path)
    grayimage1 = cv2.cvtColor(readimage1, cv2.COLOR_BGR2GRAY)
    grayimage2 = cv2.cvtColor(readimage2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(grayimage1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(grayimage2, None)
    if(type(descriptors1) == type(None) or type(descriptors2) == type(None)):
        return ret_dict
    bf = cv2.BFMatcher()
    good = []
    matches = bf.knnMatch(descriptors1,descriptors2,2)
    for m,n in matches:
        if m.distance < n.distance * 0.75:
            good.append(m)
    for i in range(len(good)):
        fir = good[i].queryIdx
        sec = good[i].trainIdx
        kpp1 = keypoints1[fir]
        kpp2 = keypoints2[sec]
    if corners != None:
        corners1 = corners[str(image1_number)]
        corners2 = corners[str(image2_number)]
        score = []
        for i in range(len(corners1)):
            score.append([])
            for j in range(len(corners2)):
                score[i].append([])
        for i in range(len(corners1)):
            for j in range(len(corners2)):
                corner1 = corners1[i]
                corner2 = corners2[j]
                low_x1 = min(corner1[0][0],corner1[3][0])
                low_y1 = min(corner1[0][1],corner1[1][1])
                high_x1 = max(corner1[1][0],corner1[2][0])
                high_y1 = max(corner1[2][1],corner1[3][1])

                low_x2 = min(corner2[0][0], corner2[3][0])
                low_y2 = min(corner2[0][1], corner2[1][1])
                high_x2 = max(corner2[1][0], corner2[2][0])
                high_y2 = max(corner2[2][1], corner2[3][1])
                m_score = 0
                for match in good:
                    k1 = keypoints1[match.queryIdx]
                    k2 = keypoints2[match.trainIdx]
                    x1 = k1.pt[0]
                    x2 = k2.pt[0]
                    y1 = k1.pt[1]
                    y2 = k2.pt[1]
                    if x1>=low_x1 and x1<=high_x1 and y1>=low_y1 and y1<=high_y1 and x2>=low_x2 and x2<=high_x2 and y2>=low_y2 and y2<=high_y2:
                        m_score+=1
                score[i][j] = m_score

    for l in range(len(corners1)):
        ret_dict[l] = None
        corner1 = corners1[l]
        num = 0
        clr = 50
        clr += 50
        color = [(255,0,0),(0,255,0),(0,0,255)]
        for j in range(len(corners2)):
            if score[l][j] > score[l][num]:
                num = j
        corner2 = corners2[num]
        for i in range(len(corner1)):
            start_point = (int(corner1[i][0]),int(corner1[i][1]))
            if i+1 < len(corner1):
                end_point = (int(corner1[i+1][0]),int(corner1[i+1][1]))
            else:
                end_point = (int(corner1[0][0]),int(corner1[0][1]))
            thickness = 2
            readimage1 = cv2.line(readimage1, start_point, end_point, color[l], thickness)
        if score[l][j] < 10:
            continue
        for i in range(len(corner2)):
            start_point = (int(corner2[i][0]),int(corner2[i][1]))
            if i+1 < len(corner2):
                end_point = (int(corner2[i+1][0]),int(corner2[i+1][1]))
            else:
                end_point = (int(corner2[0][0]),int(corner2[0][1]))
            thickness = 2
            readimage2 = cv2.line(readimage2, start_point, end_point, color[l], thickness)
        ret_dict[l] = num
    image = np.concatenate((readimage1,readimage2))
    cv2.imwrite("D:\\Data\\Compare\\SIFT\\" + str(image1_number) + '-' + str(image2_number) + '.jpg',image)
    return ret_dict

def extract_corner_from_json(json_file= None):
    load_dict = {}
    with open("D:\\Data\\vedio\\vedio.json", 'r') as load_f:
        load_dict = json.load(load_f)
    dictionary = {}
    for key in load_dict["ad_units_instances"]:
        if not key in dictionary:
            dictionary[key] = {}
        c = 0
        for box in load_dict["ad_units_instances"][key]:
            dictionary[key][c] = []
            for i in range(len(load_dict["ad_units_instances"][key][box]["unit"])):
                point = (int(load_dict["ad_units_instances"][key][box]["unit"][i][0]),int(load_dict["ad_units_instances"][key][box]["unit"][i][1]))
                dictionary[key][c].append(point)
            c += 1
    return dictionary
def extract_corner_from_ad_unit_json(json_file= r'D:\Data\source\00be30b84ff5d8a709b9fd00f001c357\ad_units.json'):
    index = 0
    load_dict = {}
    with open(json_file, 'r') as load_f:
        load_dict = json.load(load_f)
    dictionary = {}
    for key in load_dict:
        if not key in dictionary:
            dictionary[key] = {}
        for cornerss in load_dict[key]:
            corners = cornerss['corners']
            dictionary[key][index] = []
            for corner in corners:
                point = (int(corner[0]),int(corner[1]))
                dictionary[key][index].append(point)
            index += 1
    return dictionary

def runtime(dictionary_path = 'D:\\Data\\vedio\\vedio_image_with_corners'):
    path = "D:\\Data\\vedio\\"
    corners = extract_corner_from_json()
    f_type = '.jpg'
    seqs = []
    for p in Path(dictionary_path).iterdir():
        tmp = str(p).split('\\')
        tmp = tmp[len(tmp) - 1]
        tmp = tmp.split(r'.')
        tmp = tmp[len(tmp) - 2]
        seqs.append(tmp)

    def Mysort(s):
        return (int)(s)

    seqs.sort(key=Mysort)
    for i in range(len(seqs)-1):
        image1_number = seqs[i]
        image2_number = seqs[i+1]
        load_image_and_write_image_after_SIFT_Match(image1_number,image2_number,corners)


def compare_img_hist(img1, img2):
    
    # Get the histogram data of image 1, then using normalize the picture for better compare
    img1_hist = cv2.calcHist([img1], [1], None, [256], [0, 256])
    img1_hist = cv2.normalize(img1_hist, img1_hist, 0, 1, cv2.NORM_MINMAX, -1)
    img2_hist = cv2.calcHist([img2], [1], None, [256], [0, 256])
    img2_hist = cv2.normalize(img2_hist, img2_hist, 0, 1, cv2.NORM_MINMAX, -1)
 
    similarity = cv2.compareHist(img1_hist, img2_hist, cv2.HISTCMP_BHATTACHARYYA)
    return similarity

def crop_image(image, points, extend_ratio=0.1):
    #print(points)
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

    croped_img = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
    return croped_img