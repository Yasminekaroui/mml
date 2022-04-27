import os
from datetime import datetime
import argparse
import multiprocessing
import TrainingModelPt, Utils_pt
import torch
import transformers
import numpy as np
import tqdm
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pathlib import Path
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split



class multi30k_de(Dataset):
    def __init__(self, config, split, max_words=30, en=False, embed_type='cls'):  
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
        self.sentences = self.read_cap_file(path=os.path.join(config['ann_root'],cap_path),max_words= max_words)
        img_path = os.path.join(split, img_filenames[split])
        self.img_ids = self.read_text_file(os.path.join(config['ann_root'],img_path))
    
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, caption in enumerate(self.sentences):
            self.img2txt[img_id] = [img_id]
            self.txt2img[txt_id] = img_id
            txt_id += 1
        
        self.config = config
        self.embed_type = embed_type
        self.annotation_path = config["ann_root"]
        self.embeddings_path = config["embeddings_root"]

        self.embeddings = self.prepare_embeddings(split)


                
    def __len__(self):
        return len(self.sentences)


    def read_text_file(self, path):
        with open(path) as file:
            lines = [line.rstrip() for line in file]
            return lines

    def read_cap_file(self, path, max_words):
        with open(path) as file:
            lines = [line.rstrip() for line in file]
            return lines

    def __getitem__(self, index):
        return self.sentences[index], self.embeddings[index]

    def prepare_embeddings(self, split):
        """
        embeddings_path(str) = "/mnt/localdata/karoui/datasets/nlvr2/blip_text_embeddings_en/"
        """
        if self.embed_type == "cls":
             filenames = {
                "train": "features_train_cls.csv",
                "val": "features_dev_cls.csv",
                "test": "features_test_cls.csv",
            }
        elif  self.embed_type == "average":    
            filenames = {
                "train": "features_train_avg.csv",
                "val": "features_dev_avg.csv",
                "test": "features_test_avg.csv",
            }
        embeddings = None
        if split in ["train", "val", "test"]:
            df = pd.read_csv(
                os.path.join(self.embeddings_path, filenames[split]), header=None
            )
            df.drop(0, inplace=True, axis=1)
            embeddings = np.array(df).tolist()

        return embeddings

class flickr_dataset(Dataset):
    def __init__(self, config, split, embed_type):
        self.config = config
        self.embed_type = embed_type
        self.language = config["language"]  # turkish
        self.annotation_path = config["ann_root"]
        self.embeddings_path = config["embeddings_root"]
        self.ann_filenames =  {"train": f"flickr30k_train_{self.language}.json", 'val':f'flickr30k_val_{self.language}.json','test':f'flickr30k_test_{self.language}.json'}
        if self.embed_type == "cls":
             self.embeddings_filenames = {
                "train": "features_train_cls.csv",
                "val": "features_dev_cls.csv",
                "test": "features_test_cls.csv",
            }
        elif  self.embed_type == "avg":    
            self.embeddings_filenames = {
                "train": "features_train_avg.csv",
                "val": "features_dev_avg.csv",
                "test": "features_test_avg.csv",
            }
        self.embeddings = self.prepare_embeddings(split=split, filenames=self.embeddings_filenames)
        self.sentences = self.prepare_sentences(self.language, split, filenames=self.ann_filenames)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.embeddings[index]

    def prepare_embeddings(self, split, filenames):
        """
        embeddings_path(str) = "/mnt/localdata/karoui/datasets/flickr/blip_embeds/"
        """
        
        embeddings = None
        if split in ["train", "val", "test"]:
            df = pd.read_csv(
                os.path.join(self.embeddings_path, filenames[split]), header=None
            )
            df.drop(0, inplace=True, axis=1)
            embeddings = np.array(df).tolist()
        print(len(embeddings))
        return embeddings

    def prepare_sentences(self, language, split, filenames):
        """
        annotation_path(str) : "/mnt/localdata/karoui/datasets/flickr/annotations/"
        language(str) : in ['id', 'sw', 'ta', 'tr', 'zh']
        """
   
        sentences = []
        # if split == "train":
        #     annotations = json.load(
        #         open(os.path.join(self.annotation_path, filenames[split]), "r")
        #     )
        #     sentences = [ann["caption"] for ann in annotations]

        # else:
        #     annotations = json.load(
        #         open(os.path.join(self.annotation_path, filenames[split]), "r")
        #     )

        #     sentences = [item for ann in annotations for item in ann]
        annotations = json.load(
                open(os.path.join(self.annotation_path, filenames[split]), "r")
            )
        sentences = [ann["caption"] for ann in annotations]

        #print(len(sentences))    
        return sentences
def createModel(modelBase, embed_type):
    model = TrainingModelPt.SentenceModel(modelBase, embed_type)
    #tokenizer = transformers.AutoTokenizer.from_pretrained(modelBase)
    return model #, tokenizer

def init_tokenizer(tokenizer):
    '''
    BLIP init_tokenizer function
    '''
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def train(model, data_loader, device, optimizer, loss_fn, epoch, tokenizer):
    metric_logger = Utils_pt.MetricLogger(delimiter="  ")

    metric_logger.add_meter(
        "loss_train", Utils_pt.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "mae_train", Utils_pt.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 10
    model.train()

    for i, (sen, embs) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        #sen, embs = sen.to(device), embs.to(device)
        inputData = Utils_pt.batchEncode(sen, tokenizer)
        ids = inputData[0].to(device)
        att = inputData[1].to(device)
        output = model((ids,att), training=True)
        embs = torch.stack(embs, dim=0)
        targets = torch.transpose(embs, 0, 1).to(device)
        loss = loss_fn(output.float(), targets.float())
        #mae = torch.abs(output - targets).sum().data / targets.size(0)
        l1loss = torch.nn.L1Loss()
        mae = l1loss(output.float(), targets.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_train=loss.item())
        metric_logger.update(mae_train=mae.item())
    

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    
    
    return {
        k: "{:.4f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }

@torch.no_grad()
def evaluate(model, data_loader, device, tokenizer, loss_fn):
    # test
    model.eval()
    metric_logger = Utils_pt.MetricLogger(delimiter="  ")
    header = "Evaluation:"
    print_freq = 10

    results = []

    for sen, embs in metric_logger.log_every(data_loader, print_freq, header):
        inputData = Utils_pt.batchEncode(sen, tokenizer)

        ids = inputData[0].to(device)
        att = inputData[1].to(device)

        embs = torch.stack(embs, dim=0)
        targets = torch.transpose(embs, 0, 1).to(device)
        

        output = model((ids,att), training=False)
        size = targets.size(0)
        loss = loss_fn(output.float(), targets.float())
        #mae = torch.abs(output - targets).sum().data / targets.size(0)
        l1loss = torch.nn.L1Loss()
        mae = l1loss(output.float(), targets.float())
        metric_logger.meters["mae_val"].update(mae.item(), n=size)
        metric_logger.meters["loss_val"].update(loss.item(), n=size)

    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())
    
    return {
        k: "{:.4f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }

def init_experiment(args):
    '''
    init paths of an experiment to save the model and the log file
    '''
    Path(os.path.join(args.output_dir, args.dataset)).mkdir(parents=True, exist_ok=True)
    exp_name = args.exp_name + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_path = os.path.join(os.path.join(args.output_dir, args.dataset), exp_name)
    log_path = os.path.join(exp_path, "log_dir")
    os.makedirs(log_path)
    weights_path = os.path.join(exp_path, "weights")
    os.makedirs(weights_path)

    return exp_path, log_path, weights_path

def trainStudentTextEncoder(config, args):
    #tensorboard
    #add the datetime so that the folder can be unique
    exp_path, log_path, weights_path = init_experiment(args)
    writer = SummaryWriter(log_path)
    print("embeding type ", args.embed_type)

    # modelBase = 'distilbert-base-multilingual-cased'
    modelBase = config["modelBase"]
    language = config["language"]  # turkish

    # numValidationSamples = 2000
    clipEmbeddingSize = config["clipEmbeddingSize"]
    learningRate = config["learningRate"]
    batchSize = config["batchSize"]
    epochs = config["epochs"]
    device = args.device
    
    print("Initialize model")
    #model, tokenizer = createModel(modelBase, args.embed_type)
    model = createModel(modelBase, args.embed_type)
    tokenizer = init_tokenizer(config[f'tokenizer_{args.lan}'])
    model.transformer.resize_token_embeddings(len(tokenizer))

    print("prepare data")
    if args.dataset == "flickr":
        full_trainset = flickr_dataset(config, "train", args.embed_type)
        idx_train, idx_val = train_test_split(
                np.arange(len(full_trainset)),
                test_size=0.1,
                random_state=1,
            )
        trainsampler = SubsetRandomSampler(idx_train)
        valsampler = SubsetRandomSampler(idx_val)
        train_loader = DataLoader(
            full_trainset,
            batch_size=batchSize,
            sampler=trainsampler,
            pin_memory=True,
            shuffle=False,
            num_workers=0,
            collate_fn=None
        )
        val_loader = DataLoader(
            full_trainset,
            batch_size=batchSize,
            sampler=valsampler,
            pin_memory=True,
            shuffle=False,
            num_workers=0,
            collate_fn=None,
        )
    else:
        trainset = multi30k_de(config, "train", args.embed_type)
        valset = multi30k_de(config, "val", args.embed_type)
        
        train_loader = DataLoader(
            trainset,
            batchSize,
            pin_memory=True,
            shuffle=True,
            num_workers=0,
            collate_fn=None
        )
        val_loader = DataLoader(
            valset,
            batchSize,
            pin_memory=True,
            shuffle=False,
            num_workers=0,
            collate_fn=None,
        )
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
   

    model = model.to(device)
    loss = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=learningRate)

    best = 1
    best_epoch = 0
    counter = 0
    best_score = None
    patience = 20
    early_stop = False
    print("Start Training")

    for epoch in tqdm(range(0, epochs)):

        train_stats = train(model, train_loader, device, optim, loss, epoch, tokenizer)

        val_stats = evaluate(model, val_loader, device, tokenizer, loss)
   
       
        writer.add_scalar("Loss/train", float(train_stats["loss_train"]), epoch)
        writer.add_scalar("mae/train", float(train_stats["mae_train"]), epoch)
        writer.add_scalar("mae/val", float(val_stats["mae_val"]), epoch)
        writer.add_scalar("Loss/val", float(val_stats["loss_val"]), epoch)
        
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
        }
            
        if best_score is None:
            #hugging face format
            model.save_pretrained(weights_path)
            print("First model saved")
            best_score = float(val_stats["mae_val"])
            best_epoch = epoch
        elif float(val_stats["mae_val"]) > best_score:
            counter += 1
            if counter >= patience:
                early_stop = True
        else:
            model.save_pretrained(weights_path)
            print("best model saved")
            best_score = float(val_stats["mae_val"])
            best_epoch = epoch
            counter = 0
        
        if early_stop:
            print(f"early stopping at epoch{epoch}") 
            break
            


        with open(os.path.join(exp_path, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
      
    print(f"Best model saved at epoch {best_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/mnt/localdata/karoui/code/Multilingual-CLIP/src/TeacherLearning/flickr.yaml",
    )
    parser.add_argument("--output_dir", default="./experiments/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log_dir", default="/logdir/")
    parser.add_argument("--exp_name", default="test_early_stop")
    parser.add_argument("--dataset", default="flickr") # or multi30k_de

    parser.add_argument("--embed_type", default="cls") #could be avg or cls
    parser.add_argument("--lan", default="tr") #language

    args = parser.parse_args()
    
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    trainStudentTextEncoder(config, args)
    print('Finished Training')