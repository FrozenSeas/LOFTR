import demo2
from utils import *
import demo
import datetime
def runtime(dictionary_path,s,e):
    document = {}
    corners = extract_corner_from_ad_unit_json(os.path.join(dictionary_path,'ad_units.json'))
    dictionary_path = os.path.join(dictionary_path,'Frames')
    print(corners)
    seqs = e
    start = s
    img1s = []
    img2s = []
    kp1s = []
    kp2s = []
    img_ps1 = []
    img_ps2 = []
    
    for i in range(start,seqs+2):
            image1_path = os.path.join(generator_image_path_on_image_number(dictionary_path,str(i),'','.jpg'))
            image2_path = os.path.join(generator_image_path_on_image_number(dictionary_path,str(i+1),'','.jpg'))
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            img1s.append(img1)
            img2s.append(img2)
            img_ps1.append(image1_path)
            img_ps2.append(image2_path)
    demo2.Detect_result_based_on_corner(img1s,img2s,corners,start,seqs,img_ps1,img_ps2)
runtime(r'/root/LOFTR/vedios',69000,70000)
"""
for i in range(1):
    runtime(r'/root/LOFTR/vedios',i*3000,i*3000+3000)
    """
