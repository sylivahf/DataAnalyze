import fiftyone as fo
"""
  数据目录
H:/uav-data/RGBT-Tiny/
  ├─labels
  └─images
dataset.yaml格式：注意每个关键字后面需要一个空格！
  path: "H:/hf/uav-data/Drone_data_IR"
  train: ./images
  val: ./images
  # Classes
  names: 
    - drone
    - bird
"""

print(fo.list_datasets()) # 打印数据集列表
# fo.delete_dataset("RGBT-Tiny") # 如果存在同名数据集，先删除
dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset,
        yaml_path = 'H:/uav-data/RGBT-Tiny/dataset.yaml',
        shuffle=False,
        name="RGBT-Tiny")   
session = fo.launch_app(dataset)
session.wait()
dataset.persistent = True # 永久保存
## 打开本地网络地址
## http://localhost:5151/datasets/RGBT-Tiny
