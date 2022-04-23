import os
import json
import random
import jsonlines

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption


def _create_entry(item):
    entry = {
        "question_id": item["question_id"],
        "image_0": item["image_0"],
        "image_1": item["image_1"],
        "caption": item["caption"],
        "concept": item["concept"],
        "label": item["label"],
    }
    return entry


def _load_dataset(annotations_path):
    """Load entries
    """
    with jsonlines.open(annotations_path) as reader:
        # Build an index which maps image id with a list of hypothesis annotations.
        items = []
        count = 0
        for annotation in reader:
            dictionary = {}
            dictionary["image_0"] = annotation["left_img"]
            dictionary["image_1"] = annotation["right_img"]
            dictionary["question_id"] = count
            dictionary["caption"] = str(annotation["caption"])
            dictionary["label"] = int(annotation["label"])
            dictionary["concept"] = str(annotation["concept"])
            #dictionary["scores"] = [1.0]
            items.append(dictionary)
            count += 1

    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries






class marvl_dataset(Dataset):
    def __init__(self, transform, image_root, ann_root, lan, en=False):  
        '''
        image_root (string): Root directory of images 
        ann_root_path (string): directory to store the annotation file
        lan (string): id, sw, ta, tr, zh
        en(bool): Translated annotations in english 
        '''
        self.en = en
        if self.en:
            filenames = {'id':'marvl-id_gmt.jsonl','sw':'marvl-sw_gmt.jsonl','ta':'marvl-ta_gmt.jsonl', 'tr': 'marvl-tr_gmt.jsonl', 'zh':'marvl-zh_gmt.jsonl'}
        else:
            filenames = {'id':'marvl-id.jsonl','sw':'marvl-sw.jsonl','ta':'marvl-ta.jsonl', 'tr': 'marvl-tr.jsonl', 'zh':'marvl-zh.jsonl'}
        
        self.entries = _load_dataset(os.path.join(ann_root,filenames[lan]))
        #self.annotation = json.load(open(os.path.join(ann_root,filenames[lan]),'r'))
        
        self.transform = transform
        self.image_root = os.path.join(image_root,lan)

        
    def __len__(self):
        return len(self.entries)
    

    def __getitem__(self, index):    
        entry = self.entries[index]
        images_path = os.path.join(self.image_root, 'images/'+ entry["concept"])
        image0_path = os.path.join(images_path,entry["image_0"])
        image0 = Image.open(image0_path).convert('RGB')
        image0 = self.transform(image0) 
        image1_path = os.path.join(images_path,entry["image_1"])
        image1 = Image.open(image1_path).convert('RGB') 
        image1 = self.transform(image1)
        sentence = pre_caption(entry['caption'], 40)
        label = entry['label']
        question_id = entry['question_id']
        
        '''
        ann = self.annotation[index]
        
        image0_path = os.path.join(self.image_root,ann['left_img'])          
        image0 = self.transform(image0)   
        
        image1_path = os.path.join(self.image_root,ann['right_img'])                  
        image1 = self.transform(image1)          

        sentence = pre_caption(ann['caption'], 40)
        
        if ann['label']=='True':
            label = 1
        else:
            label = 0
        '''    
        words = sentence.split(' ')
        
        if 'left' not in words and 'right' not in words:
            if random.random()<0.5:
                return image0, image1, sentence, label, question_id
            else:
                return image1, image0, sentence, label, question_id
        else:
            if random.random()<0.5:
                return image0, image1, sentence, label, question_id
            else:
                new_words = []
                for word in words:
                    if word=='left':
                        new_words.append('right')
                    elif word=='right':
                        new_words.append('left')        
                    else:
                        new_words.append(word)                    
                        
                sentence = ' '.join(new_words)
                return image1, image0, sentence, label, question_id
            
            