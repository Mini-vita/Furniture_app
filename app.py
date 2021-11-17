import streamlit as st
import io
# from state import get
from streamlit.script_request_queue import RerunData
from streamlit.script_runner import RerunException
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import os
import cv2
import time
import os
import glob
import scipy.spatial.distance as distance
import re


import os.path
from numpy import dot
from numpy.linalg import norm
import requests
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import pymysql
import tensorflow as tf
from keras import Model


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#global variable
target_classess=['Cabinetry', 'Chair', 'Couch', 'Lamp', 'Table'] 
target_dict = {
    'Cabinetry': 'cabinet',
    'Chair': 'chair',
    'Couch': 'sofa',
    'Table': 'table'
}

app_dir =  '/app/'



@st.cache(allow_output_mutation=True,show_spinner=False)
def get_bbox_list(output):

    bbox_list=[]
    bbox_class_list=output["instances"].pred_classes     # get each predicted class from output, return a torch tensor
    bbox_cor_list=output["instances"].pred_boxes          # get each bounding box coordinate(xmin_ymin_xmax_ymax) from output, return a torch tensor
    bbox_class_list=bbox_class_list.cpu().numpy()  
    
    #convert coordinate to numpy
    new_list=[]
    for i in bbox_cor_list:                               
        i=i.cpu().numpy()
        new_list.append(i)
    bbox_cor_list=new_list
    #combine to a new list with dict of class and coordinate
    for i in range(len(bbox_class_list)):                
        # store each class and corresponding coordinate to dict
        temp_dict={'class':bbox_class_list[i],'coordinate':bbox_cor_list[i]}  
        bbox_list.append(temp_dict)

    return bbox_list

def save_bbox_image(read_dir, box, save_dir):
    final_img_dict=[]
    counter_=1
    img = Image.open(read_dir)
    #Convert CV2 image to PIL Image for cropping
#     img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
#     finalimg = Image.fromarray(img)

    for ind, i in enumerate(box):
        #name img file with index
        file_name=str(ind+1)+'.jpg'
        path=save_dir+ file_name
        # get bounding box coordinate for corping
        coordinate=i.get('coordinate')  
        # bbox coordinate should be in list when out put from detectron and change to tuple
        coordinate=tuple(coordinate)
        #crop image and save
        crop_img=img.crop(coordinate)
        crop_img.save(save_dir+file_name)
        #store it in a dictionary with file name and class
     
        temp_dict={'File_name':file_name,'class':target_classess[int(i['class'])]}
        final_img_dict.append(temp_dict)
        counter_+=1

    return final_img_dict

def getfurnilist(img_dict):
    f_list = []
    for index, item in enumerate(img_dict):
        f_list.append(str(index+1)+' - ' + item['class'])
    return f_list

@st.cache(allow_output_mutation=True,show_spinner=False)
#history 삭제하기
def clearold():
    files = glob.glob('/app/'+'*.jpg')
    if files:
        for f in files:
            os.remove(f)

#mariadb와 연결하기! 
@st.cache(allow_output_mutation=True,show_spinner=False, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    conn = pymysql.connect(host='34.127.31.177', user='root', password='0403', db='daewonng12', charset='utf8')
    return conn


@st.cache(allow_output_mutation=True,show_spinner=False)
#crop한 사진에 대한 feature_vector csv 불러오기
def load_feature_csv():
    feature_df = pd.read_csv('/app/style_all.csv')
    return feature_df

#img 경로를 받아 tensor로 변환하는 함수
def load_img(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    im = np.resize(im,(224,224,3))
    img = tf.convert_to_tensor(im, dtype=tf.float32)[tf.newaxis, ...] 
    return img

#input image를 resnet을 통해 feature extraction 하는 코드
def input_feature_vector(path):
    module_handle = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
    module = hub.load(module_handle)
    
    img = load_img(path)
    features = module(img)
    target_image = np.squeeze(features)
    
    return target_image 

def euclid_dist(A, B):

    return np.linalg.norm(A-B)
 
def euclid_dist_df(target_image):
    feature_df = load_feature_csv()
    euclid_dist_df = []
    
    #제품 정보가 있는 제품들의 index
    r_idx = feature_df[feature_df['info'] == True].index
    
    for idx in feature_df.index:
        vect = [float(num.replace(' ','')) for num in feature_df.iloc[idx]['feature_vectors'].replace('[', '').replace(']', '').split(',')]
        euclid_dist_df.append(euclid_dist(target_image, vect))
     

    euclid_dist_df = np.array(euclid_dist_df)
    r_euclid_dist = euclid_dist_df[r_idx] #제품 정보 0
    x_euclid_dist = np.delete(euclid_dist_df, r_idx) #제품 정보 x
    
    r_euclid_df = feature_df.iloc[r_idx].copy() #제품 정보가 있는 제품들만 가져오기! 
    x_euclid_df = feature_df.drop(r_idx, axis=0)
    
    #감성점수로 가중치 주기 
    r_euclid_df['weight'] = np.multiply(r_euclid_dist, (r_euclid_df['senti_value'].values)) #감성 점수의 가중치
    r_euclid_df.sort_values('weight', ascending=False, inplace=True)
    r_euclid_df.reset_index(drop=True, inplace=True)
    
    #감성점수가 존재 x -> 유사도만 반영하기
    x_euclid_df['weight'] = x_euclid_dist
    x_euclid_df.sort_values('weight', ascending=False, inplace=True)
    x_euclid_df.reset_index(drop=True, inplace=True)
    return r_euclid_df.head(500), x_euclid_df.head(500)

def cate_euclid_df(target_image, on):
    r_, x_ = euclid_dist_df(target_image)
    r_t = r_[r_['category'] == on]
    r_f = r_[r_['category'] != on]
    r_t.reset_index(drop=True, inplace=True)
    r_f.reset_index(drop=True, inplace=True)
    x_ = x_[x_['category'] == on]
    x_.reset_index(drop=True, inplace=True)
    
    conn = init_connection()
    
#     r_의 id와 일치하는 애들을 sql에서 조회
#    제품 정보의 여부 지정
    sql1 = "select * from finalreviewreco WHERE category = %s AND id IN %s;"
    params1 = tuple([str(ind) for ind in r_t['id'].values.tolist()])
    sql2 = "select * from nonreviewreco WHERE category = %s AND id IN %s;"
    params2 = tuple([str(ind) for ind in x_['id'].values.tolist()])
    sql3 = "select * from finalreviewreco WHERE category != %s AND id IN %s;"
    params3 = tuple([str(ind) for ind in r_f['id'].values.tolist()])

    
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute(sql1, (on, params1))
            result1 = cur.fetchall()
            cur.execute(sql2, (on, params2))
            result2 = cur.fetchall()
            cur.execute(sql3, (on, params3))
            result3 = cur.fetchall()
            rev_df = pd.DataFrame(result1)
            non_df = pd.DataFrame(result2)
            other_df = pd.DataFrame(result3)
            cur.close()
            return rev_df, non_df, other_df

            
#app start
st.set_page_config(layout="wide")

cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# Set threshold for this model
cfg.MODEL.WEIGHTS = "/app/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 
predictor = DefaultPredictor(cfg)


st.write('**_Here is where you furnish your home with a Click, just from your couch._**')
st.sidebar.image(Image.open('/app/logo_new.png'), width = 200)
st.sidebar.header("Choose a furniture image for recommendation.")


clearold()


uploaded_file = st.file_uploader("Choose an image", type=['png','jpeg','jpg'])

if uploaded_file is not None:
    #save user image and display success message
    image = Image.open(uploaded_file)
    user_img_path = app_dir+ uploaded_file.name
    image.save(user_img_path)
    
    
    st.sidebar.image(image,width = 250)

    st.sidebar.success('Upload Successful! Please wait for object detection.')
    
    furni_list = []
    with st.spinner('Working hard on finding furniture...'):
        im = cv2.imread('/app/' + uploaded_file.name)
        outputs = predictor(im)
        
        bb_list = get_bbox_list(outputs)
        
        imgdict = save_bbox_image(user_img_path, bb_list, '/app/')
        
        furni_list = getfurnilist(imgdict)
        
        if furni_list:
            for i,file in enumerate(furni_list):
                st.sidebar.write(file)
                d_image = '/app/' + str(i+1)+'.jpg'
                cropped_im = Image.open(d_image)
                st.sidebar.image(cropped_im, width=150)
        else:
            st.sidebar.write('detect fail.....')
        
            
    display = furni_list
    options = list(range(len(furni_list)))
    option = st.selectbox('Which furniture do you want to look for? ', options, format_func = lambda x: display[x])
    if st.button('Confirm to select '+ furni_list[option]):
        obj_class = target_dict[imgdict[option]['class']]
        st.write(obj_class)
        
        pred_path = '/app/' + str(option+1)+'.jpg'
        image_array = input_feature_vector(pred_path) #feature extraction

#         특정 카테고리에서 리뷰가 수집된 것들 중 유사한 스타일의 아이템 가져오기! 
        review_df, non_df, oth_df = cate_euclid_df(image_array, on=obj_class)
        
        st.write("### " + "Recommendation for: "+imgdict[option]['class']+'s')
        
#       show more html 접목해보기

        c1, c2, c3, c4, c5 = st.columns((1, 1, 1, 1, 1))
        columnli = [c1,c2,c3,c4,c5]

        for i,column in enumerate(columnli):
            coltitle = re.match(r"^([^,]*)",str(review_df[review_df['category'] == obj_class][i:i+1].item_nm.values.astype(str)[0])).group()
            colcat = str(review_df[review_df['category']==obj_class][i:i+1].category.values.astype(str)[0])
            colurl = str(review_df[review_df['category']==obj_class][i:i+1].img_url.values.astype(str)[0])
            colprice = str(review_df[review_df['category']==obj_class][i:i+1].item_price.values.astype(str)[0])
            collink = str(review_df[review_df['category']==obj_class][i:i+1].url.values.astype(str)[0])
            column.image(Image.open(requests.get(colurl, stream=True).raw),width=180)
            column.write(colprice)  
            column.write(coltitle)
            column.write("[View more product info]("+collink+")")
                         
        st.text("")
        st.write("### " + "Don't have to worry finding "+imgdict[option]['class']+" items in other's home: ")
        c6,c7,c8,c9,c10 = st.columns((1, 1, 1, 1, 1))
        columnli2 = [c6,c7,c8,c9,c10]
                         
                         
        for i,column in enumerate(columnli2):
            colcat = str(non_df[i:i+1].category.values.astype(str)[0])
            colurl = str(non_df[i:i+1].img_url.values.astype(str)[0])
            collink = str(non_df[i:i+1].url.values.astype(str)[0])
            column.image(Image.open(requests.get(colurl, stream=True).raw),width=180)
            column.write("[View more product info]("+collink+")")
            
        st.text("")
        st.write("### " + "Some other non-"+imgdict[option]['class']+"s items you may like: ")
        c11,c12,c13,c14,c15 = st.columns((1, 1, 1, 1, 1))
        columnli3 = [c11,c12,c13,c14,c15]
                         
                         
        for i,column in enumerate(columnli3):
            coltitle = re.match(r"^([^,]*)",str(oth_df[i:i+1].item_nm.values.astype(str)[0])).group()
            colcat = str(oth_df[i:i+1].category.values.astype(str)[0])
            colurl = str(oth_df[i:i+1].img_url.values.astype(str)[0])
            colprice = str(oth_df[i:i+1].item_price.values.astype(str)[0])
            collink = str(oth_df[i:i+1].url.values.astype(str)[0])
            column.image(Image.open(requests.get(colurl, stream=True).raw),width=180)
            column.write(colcat)
            column.write(colprice)  
            column.write(coltitle)
            column.write("[View more product info]("+collink+")") 
