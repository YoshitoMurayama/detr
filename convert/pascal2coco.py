import numpy as np
import os, sys
import pathlib
from lxml import etree
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import re
import itertools
import cv2

import label_names

PASCAL_ROOT = ['/mnt/data/train', '/mnt/data/valid']

def imread(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".npy":
        return np.load(file_path)
    else:
        return cv2.imread(file_path)

def glob_images(root):
    """
        glob all image path under "root" folder
    """    
    ext_filter = '.*\.(jpg|jpeg|png|bmp|tif|npy)$'

    pobj    = pathlib.Path(root)
    #path    = os.path.join(root, '**/*')
    files    = [str(fp) for fp in pobj.glob('**/*')]
    #files   = glob.glob(root, recursive=True)
    files   = [f for f in files if re.search(ext_filter, f, re.IGNORECASE)]
    return files

def post_process_health(data_list):
    """
        dataset specific post processing
    """

    def parse_health_certificate_label(label):
        suffix2code = {'p1':1, 'p2':2}

        data    = label.split('-')
        name    = data[0]

        if not name in label_names.ALL_CLASSES:
            return None

        if 1 < len(data):
            suffix = data[1]

            if not suffix in suffix2code.keys():
                return None

            suffix = suffix2code[suffix]

        else:
            suffix = 0

        return name, suffix


    dst = []

    for data in data_list:

        label = data['category_id']
        label = parse_health_certificate_label(label)

        if None is label:
            continue

        data['category_id'] = label[0]
        data['attribute_id'] = label[1]

        dst.append(data)

    return dst

def load_pascal_xml(xml_path):
    """
        load pascal style xml as list of dictionary
        one dict for each bbox in xml, each dict has file name of xml as ID
    """

    TEMPLATE = {
        'segmentation': [],
        'area': 0,              # w * h of bbox
        'iscrowd': 0,           # ?
        'image_id': None,       # uniq number for each image / str fname here
        'bbox': [],             # x, y, w, h
        'category_id': None,    # uniq number for each label name / str label name here
        'id': None,             # uniq number for each bbox / dont set here
        'attribute_id': 0,      # health data specific attribute / number id after post processing
    }

    dst     = []
    tree    = etree.parse(xml_path)
    id_name = os.path.splitext(os.path.basename(xml_path))[0]

    for obj in tree.xpath('//object'):
        bbox = TEMPLATE.copy()
        bbox['image_id'] = id_name
 
        x1 = float(obj.xpath(".//xmin")[0].text)
        y1 = float(obj.xpath(".//ymin")[0].text)
        x2 = float(obj.xpath(".//xmax")[0].text)
        y2 = float(obj.xpath(".//ymax")[0].text)
        hh = x2-x1
        ww = y2-y1

        bbox['bbox'] = [x1, y1, hh, ww]
        bbox['area'] = hh*ww
        bbox['segmentation'] = [[x1,y1, x1,y2, x2,y2, x2,y1]] # point list of object contour polygon
        bbox['category_id'] = obj.xpath(".//name")[0].text
        bbox['xmlpath'] = xml_path

        dst.append(bbox)

    # health certificate specific post processing
    dst = post_process_health(dst)

    return dst

def assign_id(annotations, image_id_offset=0, id_offset=0):
    """assign id number for each bbox
        image_id / id is assigned as list order
        convert str category_id to number

    Args:
        annotations (list(list(dic))): axis order [image, bbox]
        image_id_offset (int): base offset for image id
        id_offset (int): base offset for id
    """
    _image_id   = image_id_offset   # uniq for image
    _id         = id_offset         # uniq for bbox

    for ii in range(len(annotations)): # "image" loop
        
        for jj in range(len(annotations[ii])): # "bbox" loop

            str_cat = annotations[ii][jj]['category_id']
            cat_id  = label_names.ALL_CLASSES.index(str_cat)

            annotations[ii][jj]['image_id']     = _image_id
            annotations[ii][jj]['id']           = _id
            annotations[ii][jj]['category_id']  = cat_id

            _id += 1

        _image_id   += 1

    return annotations

def get_image_info(args):
    TEMPLATE = {
        'license': 0,
        'file_name': '',
        'coco_url': '',
        'height': None,
        'width': None,
        'data_caputed': None,
        'flickr_url': '',
        'id': None
    }

    image_path, image_id = args

    dst = TEMPLATE.copy()
    img = imread(image_path)
    hh, ww = img.shape[:2]
    
    dst['file_name']    = os.path.abspath(image_path)
    dst['height']       = hh
    dst['width']        = ww
    dst['id']           = image_id

    return dst

def build_image_info(image_path_list):

    args = [(image_path, ii) for ii, image_path in enumerate(image_path_list)]

    with ProcessPoolExecutor(max_workers=10) as executer:
        image_info = list(tqdm(executer.map(get_image_info, args), total=len(args)))

    return image_info

def build_cat_info(all_classes):
    TEMPLATE = {
        'supercategory': '',
        'id': None,
        'name': '',
    }

    dst = []

    for ii, name in enumerate(all_classes):
        _info = TEMPLATE.copy()
        _info['supercategory']  = name
        _info['id']             = ii
        _info['name']           = name

        dst.append(_info)

    return dst

def glob_data_pairs(root):

    image_path_list = glob_images(root)
    image_path_list = [img_path for img_path in image_path_list if os.path.exists(os.path.splitext(img_path)[0] + '.xml')]
    xml_path_list = [os.path.splitext(img_path)[0] + '.xml' for img_path in image_path_list]
    
    return image_path_list, xml_path_list

if __name__ == '__main__':

    image_id_offset = 0
    id_offset       = 0

    for pp, root in enumerate(PASCAL_ROOT):

        image_path_list, xml_path_list = glob_data_pairs(root)

        print('loading xml')
        annotations = [load_pascal_xml(xml_path_list[ii]) for ii in tqdm(range(len(xml_path_list)))]
        annotations = assign_id(annotations)
        # flatten [image, bbox] 2d_list to [image*bbox] 1d_list
        annotations = list(itertools.chain.from_iterable(annotations))

        print('build image info')
        image_info  = build_image_info(image_path_list)
        cat_info    = build_cat_info(label_names.ALL_CLASSES)

        master = {
            "info": {},
            'licenses': [],
            'images': image_info,
            'annotations' : annotations,
            'categories': cat_info   
        }

        print('writing json')
        with open('health_{}_20200727.json'.format(pp), 'w') as fp:
            fp.write(json.dumps(master, ensure_ascii=False, indent=4))

        print('done.')

        image_id_offset = len(image_path_list)
        id_offset = len(annotations)