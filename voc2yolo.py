## voc标注转yolo
import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm

class_names = ['drone', 'plane', 'pedestrian','ship','car','cyclist','bus']
os.makedirs("H:/uav-data/RGBT-Tiny/labels",exist_ok=True)

for filename in tqdm(os.listdir("H:/uav-data/RGBT-Tiny/annotations_voc")):
    tree = ET.parse(os.path.join("H:/uav-data/RGBT-Tiny/annotations_voc",filename))
    root = tree.getroot()
    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)
    annos = []
    for obj in root.iter("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        x_center = (xmin + xmax) / 2 / W
        y_center = (ymin + ymax) / 2 / H
        w = (xmax - xmin) / W
        h = (ymax - ymin) / H
        name_id = class_names.index(name) 
        # x_center,y_center,w,h 保留小数点6位
        # w 转为 小数点6位
        w = float(f"{w:.6f}")
        h = float(f"{h:.6f}")
        x_center = float(f"{x_center:.6f}")
        y_center = float(f"{y_center:.6f}") 
        annos.append(f"{name_id} {x_center} {y_center} {w} {h}")
    with open(os.path.join("H:/uav-data/RGBT-Tiny/labels",filename.replace(".xml",".txt")),"w") as f:
        f.write("\n".join(annos))
##fiftyone 加载图像、标注
