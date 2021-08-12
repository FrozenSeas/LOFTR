import demo2
from utils import *
import demo
def runtime(dictionary_path):
    document = {}
    corners = extract_corner_from_ad_unit_json(os.path.join(dictionary_path,'ad_units.json'))
    dictionary_path = os.path.join(dictionary_path,'ad_units')
    print(corners)
    seqs = 2500
    """
    img1s = []
    img2s = []
    kp1s = []
    kp2s = []
    for i in range(1500,seqs-1):
            image1_path = os.path.join(generator_image_path_on_image_number(dictionary_path,str(i),'','.jpg'))
            image2_path = os.path.join(generator_image_path_on_image_number(dictionary_path,str(i+1),'','.jpg'))
            img1,img2,kp1,kp2 = demo2.Detect_process(image1_path,image2_path)
            demo.Detect_result_based_on_corner(img1,img2,kp1,kp2,corners,i)
            img1s.append(img1)
            img2s.append(img2)
            kp1s.append(kp1)
            kp2s.append(kp2)
    """
    img1s = np.load('img1s_1500_2500.npy', allow_pickle=True)
    kp1s = np.load('kp1s_1500_2500.npy', allow_pickle=True)
    img2s = np.load('img2s_1500_2500.npy', allow_pickle=True)
    kp2s = np.load('kp2s_1500_2500.npy', allow_pickle=True)
    
    demo2.Detect_result_based_on_corner(img1s,img2s,kp1s,kp2s,corners,seqs)
    
runtime(r'D:\LoTFR\LOFTR\vedios\024451a9e714dac5233751fee40d5ab3')