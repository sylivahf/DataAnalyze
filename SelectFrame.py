import fiftyone.zoo as foz
import fiftyone.brain as fob
import numpy as np
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import torch    

# resnet50生成特征向量
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def extract_embeddings(img_path):
    input_tensor = transform(Image.open(img_path).convert("RGB")) 
    with torch.no_grad():
        embedding = model(input_tensor.unsqueeze(0)).squeeze(0).numpy()
    return embedding

# dataset2.delete_sample_field('embedding') # 如果需要删除某个字段

if __name__ == '__main__':
  embeddings = [] # 存储所有样本的embedding特征
  for i,sample in enumerate(dataset2): # dataset2 为fiftyone格式数据集
    filepath = sample['filepath']
    idx = os.path.basename(filepath).split('_')[:3]
    idx = '_'.join(idx)
     
    embedding = extract_embeddings(sample['filepath'])
    # sample['embedding'] = embedding # 如果fiftyone格式数据增加字段 embedding
    embeddings.append(embedding)
  
  ## 基于某帧，筛选差异较大帧(帧序中关键帧)
  diff_dir = 'H:\\hf\\uav-data\\RGBT-Tiny-keyframe'
  os.makedirs(diff_dir, exist_ok=True)
  thresh = 0.005
  for i,sample in enumerate(dataset2):
    filepath = sample['filepath']
    idx = os.path.basename(filepath).split('_')[:3]
    idx = '_'.join(idx)
  
    if i == 0:
        name = os.path.basename(filepath)
        os.system('cp {} {}'.format(filepath, os.path.join(diff_dir,name)))
        prev_embedding = embeddings[i]
        continue
    
    if idx == 'DJI_0173_2':
        thresh = 0.018
    cur_embedding = embeddings[i]
    cos_sim = np.dot(prev_embedding.squeeze(), cur_embedding.squeeze()) / (np.linalg.norm(prev_embedding.squeeze()) * np.linalg.norm(cur_embedding.squeeze()) + 1e-8)
    diff_score = 1 - cos_sim
    if diff_score > thresh: # 默认0.2
        name = os.path.basename(filepath)
        os.system('cp {} {}'.format(filepath, os.path.join(diff_dir,name)))
        prev_embedding = cur_embedding
  
  
