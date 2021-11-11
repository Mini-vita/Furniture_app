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
import urllib.request

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
app_dir =  'mini-vita/furniture_app/main/'
save_img_dir = 'mini-vita/furniture_app/main/app/detect/'
train_path='mini-vita/furniture_app/main/'
test_path='mini-vita/furniture_app/main/app/test/'
valid_path='mini-vita/furniture_app/main/app/validation/'
# dfpath = 'ikeadata/ikea_final_model0.csv'

#mariadb와 연결하기! 
@st.cache(allow_output_mutation=True,show_spinner=False, hash_funcs={"_thread.RLock": lambda _: None})
def init_connection():
    conn = pymysql.connect(host='34.127.31.177', user='root', password='0403', db='daewonng12', charset='utf-8-sig')
    return conn

@st.cache(ttl=600)
def run_query(query, param):
    with conn.cursor() as cur:
        cur.execute(query, param)
        return cur.fetchall()

#detectron2 
# @st.cache(allow_output_mutation=True,show_spinner=False)
# def load_ikeadf(path):
#     return pd.read_pickle(path)
     
#Initialize the detectron model
@st.cache(allow_output_mutation=True,show_spinner=False)
def load_json_labels(image_folder):
  from detectron2.structures import BoxMode
   # """
   # Returns Detectron2 style labels of images in image_folder based on JSON label file in image_folder.
    
    #TODO -- Maybe create some verbosity here? AKA, what are the outputs?
    #TODO -- what if annotations = None? Can we create a call to create an annotations CSV in 1 hit?
    
    #Params
   # ------
   # image_folder (str): target folder containing images
    #"""
    # Get absolute path of JSON label file
  for file in os.listdir(image_folder):
    if file.endswith(".json"):
      json_file = os.path.join(image_folder, file)

  # TODO: Fix this assertion
  assert json_file, "No .json label file found, please make one with annots_to_json()"

  with open(json_file, "r") as f:
    img_dicts = json.load(f)

  # Convert bbox_mode to Enum of BoxMode.XYXY_ABS (doesn't work loading normal from JSON)
  for img_dict in img_dicts:
    for annot in img_dict["annotations"]:
      annot["bbox_mode"] = BoxMode.XYXY_ABS

  return img_dicts


def register_datasets(train_p, test_p, valid_p=None, target_classes=None):
  """
  Registers a Detectron2 style dataset from training paths.
  Params
  ------
  train_path (str) : pathname to training data containing training images
  valid_path (str) : pathname to validation data containing validation images
  """
  # TODO - update to accept any kind of path, e.g. not only coffeemaker, maybe could take a dict as input?
  # E.g. {"training": "path/to/training",
  #          "valid": "path/to/valid"}
  for d in [train_p,test_p, valid_p]:
    dataset_name = d.split("/")[-1]
    #print("Registering: {}".format(dataset_name))
    DatasetCatalog.register(dataset_name, lambda d=d: load_json_labels(d))
    MetadataCatalog.get(dataset_name).set(thing_classes=target_classes)
  return MetadataCatalog.get(dataset_name)

def model_load():
    if not 'model_final.pth' in os.listdir('.'):
         with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                url = 'https://www.dropbox.com/s/9an2i3twogsda7l/model_final.pth?dl=1'
                r = requests.get(url, allow_redirects=True)
                model = pickle.loads(r.content)
                model = torch.load(model)
         return model
    
def initialization():
    DatasetCatalog.clear()
    furniture_data_metadata= register_datasets(train_p=train_path, test_p=test_path, valid_p=valid_path, target_classes=target_classess)
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    # Set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    
    #우리가 만든 모델로 가중치 가져오기! 
    cfg.MODEL.WEIGHTS = model_load()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 
    predictor = DefaultPredictor(cfg)
    return cfg, predictor



#Get a list of bounding box item with class
def get_bbox_list(outputs):

    bbox_list=[]
    bbox_class_list=outputs["instances"].pred_classes     # get each predicted class from output, return a torch tensor
    bbox_cor_list=outputs["instances"].pred_boxes          # get each bounding box coordinate(xmin_ymin_xmax_ymax) from output, return a torch tensor
    bbox_class_list=bbox_class_list.cpu().numpy()         # convert class to numpy 
    
    #conver coordinate to numpy
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


def save_bbox_image(read_dir,bbox_list,save_dir):
    img_dict=[]
    counter=1
    img = Image.open(read_dir)
    #Convert CV2 image to PIL Image for cropping
#     img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
#     finalimg = Image.fromarray(img)

    for i in bbox_list:
        #name img file with index
        file_name=str(counter)+'.jpg'
        path=save_dir+ file_name
        # get bounding box coordinate for corping
        coordinate=i.get('coordinate')  
        # bbox coordinate should be in list when out put from detectron and change to tuple
        coordinate=tuple(coordinate)
        #crop image and save
        crop_img=img.crop(coordinate)
        crop_img.save(path)
        #store it in a dictionary with file name and class
        target_class=['Cabinetry', 'Chair', 'Couch', 'Lamp', 'Table'] 
        temp_dict={'File_name':file_name,'class':target_class[int(i['class'])]}
        counter+=1
        img_dict.append(temp_dict)

    return img_dict

@st.cache(allow_output_mutation=True,show_spinner=False)
#crop한 사진에 대한 feature_vector csv 불러오기
def load_feature_csv():
    url = "https://www.dropbox.com/s/pbfco5zjbhe2aw7/style_all.csv?dl=1"  # dl=1 is important
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()
    return data

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
    target_image = np.sqeeze(features)
    
    return target_image 

def euclid_dist(A, B):

    return np.linalg.norm(A-B)
 
def euclid_dist_df(target_image):
    feature_df = load_feature_csv()
    euclid_dist_df = []
    
    #제품 정보가 있는 제품들의 index
    r_idx = feature_df[feature_df['info'] == True].index
    
    for idx in feature_df.index:
        vect = [float(num) for num in feature_df.iloc[idx]['feature_vectors']]
        euclid_dist_df.append(euclid_dist(target_image, vect))

    euclid_dist_df = np.array(euclid_dist_df)
    r_euclid_dist = euclid_dist_df[r_idx] #제품 정보 0
    x_euclid_dist = np.delete(euclid_dist_df, r_idx) #제품 정보 x
    
    r_euclid_df = feature_df[r_idx].copy() #제품 정보가 있는 제품들만 가져오기! 
    x_euclid_df = feature_df.drop(r_idx)
    
    #감성점수로 가중치 주기 
    r_euclid_df['weight'] = np.multiply(r_euclid_dist, (feature_df['senti_value'].values)) #감성 점수의 가중치
    r_euclid_df.sort_values('weight', inplace=True).reset_index(drop=True, inplace=True)
    
    #감성점수가 존재 x -> 유사도만 반영하기
    x_euclid_df['weight'] = x_euclid_dist
    return r_euclid_df, x_euclid_df

def cate_euclid_df(target_image, on, how = True, rev=True):
    r_, x_ = euclid_dist_df(target_image)
    if how == True: #특정 카테고리 것
        r_ = r_[r_['category'] == on]
    else: #특정 카테고리 말고 (의자 -> 유사 디자인의 책상 추천)
        r_ = r_[~(r_['category'] == on)]
    r_.reset_index(drop=True, inplace=True)
    x_ = x_[x_['category'] == on]
    x_.reset_index(drop=True, inplace=True)
    
    
#     r_의 id와 일치하는 애들을 sql에서 조회
#    제품 정보의 여부 지정
    if rev == True:
        sql = "select * from final_review_reco where id IN (%s)"
        params = r_['id'].values.tolist()

    else:
        sql = "select * from non_review_reco where id IN (%s)" 
        params = x_['id'].values.tolist()

    result = run_query(sql, params)
    reco_df = pd.DataFrame(result)
    return reco_df


#history 삭제하기
def clearold():
    files = glob.glob('./app/detect/'+'*.jpg')
    if files:
        for f in files:
            os.remove(f)

            
@st.cache(show_spinner=False)
def GetImage(user_img_path):
    # cfg, predictor = initialization()
    #delete unncessay history file
#     files = glob.glob(save_img_dir+'*.jpg')
#     if files:
#         for f in files:
#             os.remove(f)

    imagecv = cv2.imread(user_img_path)
    outputs = predictor(imagecv)
    bbox_list=get_bbox_list(outputs) # get each bbx info
    final_img_dict = save_bbox_image(user_img_path, bbox_list,save_img_dir)
    #get a furniture option list which is well formatted for users to read
    

    return bbox_list,final_img_dict

def getfurnlist(img_dict):
    furniture_list = []
    for index, item in enumerate(img_dict):
        furniture_list.append(str(index+1)+' - '+item['class'])

    return furniture_list


#app start
st.set_page_config(layout="wide")
cfg, predictor = initialization()
# model = load_similarity_model()

#st.image(Image.open("idecor_logo.png"), width = 700)
st.write('**_Here is where you furnish your home with a Click, just from your couch._**')
st.sidebar.header("Choose a furniture image for recommendation.")

#load the model


clearold()

uploaded_file = st.file_uploader("Choose an image with .jpg format", type="jpg")

if uploaded_file is not None:
    #save user image and display success message
#     ikeadf = load_ikeadf(dfpath)
    
    image = Image.open(uploaded_file)
    user_path = app_dir+uploaded_file.name
    image.save(user_path)
    
    st.sidebar.image(image,width = 250)

    st.sidebar.success('Upload Successful! Please wait for object detection.')

    #get image list from the detectron model
    furniturelist=[]
    with st.spinner('Working hard on finding furniture...'):
        #delete unncessay history file


        bbli,imgdict= GetImage(user_path)
        furniturelist = getfurnlist(imgdict)
        #open cropped image of furniture
        for i,file in enumerate(furniturelist):
            st.sidebar.write(file)
            d_image = save_img_dir+str(i+1)+'.jpg'
            st.sidebar.image(Image.open(d_image),width = 150)
            
    #provide select box for selection
    display = furniturelist
    options = list(range(len(furniturelist)))
    option = st.selectbox('Which furniture do you want to look for?', options, format_func=lambda x: display[x])
    
    if st.button('Confirm to select '+furniturelist[option]):
        obj_class = target_dict[imgdict[option]['class']]
        pred_path = save_img_dir+str(option+1)+'.jpg'
        image_array = input_feature_vector(pred_path) #feature extraction

#         특정 카테고리에서 리뷰가 수집된 것들 중 유사한 스타일의 아이템 가져오기! 
        df = cate_euclid_df(image_array, on=obj_class, how=True, rev=True)
        
        st.write("Recommendation for: "+imgdict[option]['class']+'s')
        
#       show more html 접목해보기

        c1, c2, c3, c4, c5 = st.beta_columns((1, 1, 1, 1, 1))
        columnli = [c1,c2,c3,c4,c5]

        for i,column in enumerate(columnli):
            coltitle = re.match(r"^([^,]*)",str(df[df['category'] == obj_class][i:i+1].item_nm.values.astype(str)[0])).group()
            colcat = str(df[df['category']==obj_class][i:i+1].category.values.astype(str)[0])
            colurl = str(df[df['category']==obj_class][i:i+1].img_url.values.astype(str)[0])
            colprice = '\\' + str(df[df['category']==obj_class][i:i+1].item_price.values.astype(str)[0])
            collink = str(df[df['category']==obj_class][i:i+1].url.values.astype(str)[0])
#             colurl = 'ikeadata/'+colcat+'/'+colpic+'.jpg'
            column.image(Image.open(colurl),width=180)
            column.write('### '+colprice)  
            column.write('##### '+coltitle)
            column.write("##### "+"[View more product info]("+collink+")")
#             column.write("[!["+coltitle+"]("+colurl+")]("+collink+")")

        st.text("")
        
#         특정 카테고리에서 리뷰가 수집된 것들 중 유사한 스타일의 아이템 가져오기! 
        df_ot = cate_euclid_df(image_array, on=obj_class, how=False, rev=True)
        #당신이 첫 구매자 #리뷰가 없는 item들 중 스타일 유사한 애들 추천하기! 
        #이 때는 가격 정보 빼고! 추천하기! 
        st.write("Some other non-"+imgdict[option]['class']+"s items you may like: ")

        c6,c7,c8,c9,c10 = st.beta_columns((1, 1, 1, 1, 1))
        columnli2 = [c6,c7,c8,c9,c10]
        for i,column in enumerate(columnli2):
            coltitle = re.match(r"^([^,]*)",str(df_ot[i:i+1].item_name.values.astype(str)[0])).group()
            colcat = str(df_ot[i:i+1].category.values.astype(str)[0])
            colurl = str(df_ot[i:i+1].img_url.values.astype(str)[0])
            colprice = '\\' + str(df_ot[i:i+1].item_price.values.astype(str)[0])
            collink = str(df_ot[i:i+1].url.values.astype(str)[0])
            column.image(Image.open(colurl),width=180)
            column.write('### '+colprice)  
            column.write('##### '+coltitle)
            column.write("##### "+"[View more product info]("+collink+")")
            
#           리뷰가 부족한 애들 중 아이템 가져오기!
        df_rx = cate_euclid_df(image_array, on=obj_class, how=True, rev=False)
        #당신이 첫 구매자 #리뷰가 없는 item들 중 스타일 유사한 애들 추천하기! 
        #이 때는 가격 정보 빼고! 추천하기! 
        st.write("Don't have to worry finding"+imgdict[option]['class']+" items in other's home: ")

        c11,c12,c13,c14,c15 = st.beta_columns((1, 1, 1, 1, 1))
        columnli3 = [c11,c12,c13,c14,c15]
        for i,column in enumerate(columnli3):
            colcat = str(df_rx[i:i+1].category.values.astype(str)[0])
            colurl = str(df_rx[i:i+1].img_url.img_url.astype(str)[0])
            collink = str(df_rx[i:i+1].url.values.astype(str)[0])
            column.image(Image.open(colurl),width=180)
            column.write("##### "+"[View more product info]("+collink+")")
