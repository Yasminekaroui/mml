
import os
import json

from torch.utils.data import Dataset
from PIL import Image

from data.utils import pre_caption
### remove it afterwards
import yaml
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode



class multi30k_retrieval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30, en=False):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory where the caption files
        split (string): val or test
        '''
        self.en = en
        if self.en:
            caption_filenames = {'train':'train.en', 'val':'val.en','test':'test.en'}
            
        else:
            caption_filenames = {'train':'train.de', 'val':'val.de','test':'test.de'}

        img_filenames = {'train':'train_images.txt', 'val':'val_images.txt', 'test':'test_images.txt'}
        cap_path = os.path.join(split, caption_filenames[split])
        self.caption = self.read_cap_file(path=os.path.join(ann_root,cap_path),max_words= max_words)
        img_path = os.path.join(split, img_filenames[split])
        self.img_ids = self.read_text_file(os.path.join(ann_root,img_path))
        self.transform = transform
        self.image_root = image_root

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, caption in enumerate(self.caption):
            self.img2txt[img_id] = [img_id]
            self.txt2img[txt_id] = img_id
            txt_id += 1




                
    def __len__(self):
        return len(self.caption)
    
    def __getitem__(self, index):    
        caption = self.caption[index]
        image_path = os.path.join(self.image_root, self.img_ids[index])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return caption, image, index 

    def read_text_file(self, path):
        with open(path) as file:
            lines = [line.rstrip() for line in file]
            return lines

    def read_cap_file(self, path, max_words):
        with open(path) as file:
            lines = [pre_caption(line.rstrip(),max_words) for line in file]
            return lines
            

def main(config):
    min_scale=0.5
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),    
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
    train_dataset = multi30k_retrieval(transform=transform_train, image_root=config['image_root'], ann_root=config['ann_root'], split='train', en=config['eng']) 
    val_dataset = multi30k_retrieval(transform=transform_test, image_root=config['image_root'], ann_root=config['ann_root'], split='val', en=config['eng']) 
    test_dataset = multi30k_retrieval(transform=transform_test, image_root=config['image_root'], ann_root=config['ann_root'], split='test', en=config['eng'] )          
      

    
    print('train', len(train_dataset)) 
    print('val', len(val_dataset))
    print('test', len(test_dataset))
    print('item', train_dataset[0]) 



if __name__ == '__main__':
    config_path = '/home/karoui/marvl_project/BLIP/configs/retrieval_multi30k.yaml'
   
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    main(config)
