import os.path as osp

# 画像データとアノテーションデータのパスを取得する関数
def make_filepath_list(rootpath):
    # 画像ファイルとアノテーションファイルへのパスのテンプレートを作成
    imgpath_templete = osp.join(rootpath, "JPEGImages", "%s.jpg")
    annopath_templete = osp.join(rootpath, "Annotations", "%s.xml")

    # 訓練と検証、それぞれのファイルのID（ファイル名）を取得する
    train_id_names = osp.join(rootpath, "ImageSets/Main/train.txt")
    val_id_names = osp.join(rootpath, "ImageSets/Main/val.txt")

    # 訓練データの画像ファイルとアノテーションファイルへのパスを保存するリスト
    train_img_list = list()
    train_anno_list = list()
    
    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (imgpath_templete % file_id)
        anno_path = (annopath_templete % file_id)

        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    val_img_list = list()
    val_anno_list = list()
    
    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (imgpath_templete % file_id)
        anno_path = (annopath_templete % file_id)

        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list
    


# バウンディングボックスの座標と正解ラベルをリスト化するクラス

import xml.etree.ElementTree as ElementTree
import numpy as np

class GetBBoxAndLabel(object):
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):
        annotation = []
        xml = ElementTree.parse(xml_path).getroot()

        for obj in xml.iter("object"):
            difficult = int(obj.find("difficult").text)
            if difficult == 1:
                continue
            bndbox = []
            name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")

            grid = ["xmin", "ymin", "xmax", "ymax"]

            for gr in (grid):
                axis_value = int(bbox.find(gr).text) - 1
                if gr == "xmin" or "xmax":
                    axis_value /= width
                else:
                    axis_value /= height
                bndbox.append(axis_value)
            
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            annotation += [bndbox]

        return np.array(annotation)



