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
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import yaml
from tqdm import tqdm



class mNLVR2_dataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.language = config["language"]  # turkish
        self.annotation_path = config["ann_root"]
        self.embeddings_path = config["embeddings_root"]

        self.embeddings = self.prepare_embeddings(split)
        self.sentences = self.prepare_sentences(self.language, split)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.embeddings[index]

    def prepare_embeddings(self, split):
        """
        embeddings_path(str) = "/mnt/localdata/karoui/datasets/nlvr2/blip_text_embeddings_en/"

        """
        filenames = {
            "train": "features_train.csv",
            "val": "features_dev.csv",
            "test": "features_test.csv",
        }
        embeddings = None
        if split in ["train", "val", "test"]:
            df = pd.read_csv(
                os.path.join(self.embeddings_path, filenames[split]), header=None
            )
            df.drop(0, inplace=True, axis=1)
            embeddings = np.array(df).tolist()

        return embeddings

    def prepare_sentences(self, language, split):
        """
        annotation_path(str) : "/mnt/localdata/karoui/datasets/nlvr2/annotations/"
        language(str) : in ['id', 'sw', 'ta', 'tr', 'zh']

        """
        filenames = {
            "train": f"nlvr_train_{language}.json",
            "val": f"nlvr_dev_{language}.json",
            "test": f"nlvr_test_{language}.json",
        }
        sentences = []
        if split in ["train", "val", "test"]:
            annotations = json.load(
                open(os.path.join(self.annotation_path, filenames[split]), "r")
            )
            sentences = [ann["sentence"] for ann in annotations]

        return sentences


########################################################################################################
def prepareDataset(tokenizer, config):
    # This part you need to prepare yourself!
    # What is needed here is a list of sentences in whatever language(s) you are interested in
    # and a matching set of Clip-Text encoder embeddings for the English counter part.

    # Pre-computed CLIP-Text encoder embeddings for 2 Million images, can be found here:
    # https://drive.google.com/drive/folders/1I9a7naSZubUATWzLFv61DQMWyFlF7wR5

    # inSents, embs = shuffleData(sentences, emeddings)  # Shuffle before selecting validation data
    trainset = mNLVR2_dataset(config, "train")
    valset = mNLVR2_dataset(config, "val")
    trainSents, trainEmbs = trainset.sentences, trainset.embeddings
    # trainData = Utils_pt.batchEncode(trainSents, tokenizer)
    evalSents, evalEmbs = valset.sentences, valset.embeddings

    # evalSents, evalEmbs = inSents[:numValidationSamples], embs[:numValidationSamples]
    evalIds, evalAtt = Utils_pt.batchEncode(evalSents, tokenizer)
    evalInData, evalLabels = (evalIds, evalAtt), torch.from_numpy(np.array(evalEmbs))
    print("Number of training samples:", len(trainSents))
    print("Number of validation samples:", len(evalSents))

    return trainSents, trainEmbs, evalInData, evalLabels


def shuffleData(sents, embs):
    shuffleOrder = np.random.choice(range(len(sents)), len(sents), replace=False)
    f = lambda x: [x[i] for i in shuffleOrder]
    return f(sents), f(embs)


def createModel(modelBase, clipEmbeddingSize):
    model = TrainingModelPt.SentenceModelWithLinearTransformation(
        modelBase, clipEmbeddingSize
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelBase)
    return model, tokenizer


def train(model, data_loader, device, optimizer, loss_fn, epoch, tokenizer):
    metric_logger = Utils_pt.MetricLogger(delimiter="  ")

    metric_logger.add_meter(
        "loss", Utils_pt.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 10

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
        loss = loss_fn(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.4f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


@torch.no_grad()
def evaluate(model, data_loader, device, tokenizer):
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

        mae = torch.abs(output - targets).sum().data / targets.size(0)

        metric_logger.meters["mae"].update(mae.item(), n=targets.size(0))
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.4f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


def trainStudentTextEncoder(config, args):
    # modelBase = 'distilbert-base-multilingual-cased'
    Utils_pt.init_distributed_mode(args)
    modelBase = config["modelBase"]
    language = config["language"]  # turkish
    annotation_path = config["ann_root"]
    embeddings_path = config["embeddings_root"]

    # numValidationSamples = 2000
    clipEmbeddingSize = config["clipEmbeddingSize"]
    learningRate = config["learningRate"]
    batchSize = config["batchSize"]
    epochs = config["epochs"]
    device =  torch.device(args.device)

    print("Initialize model")

    model, tokenizer = createModel(modelBase, clipEmbeddingSize)
    print("prepare data")

    trainSents, trainEmbs, evalIn, evalLabels = prepareDataset(tokenizer, config)
    trainset = mNLVR2_dataset(config, "train")
    valset = mNLVR2_dataset(config, "val")
    if args.distributed:
        num_tasks = Utils_pt.get_world_size()
        global_rank = Utils_pt.get_rank() 
        sampler_train = torch.utils.data.DistributedSampler(trainset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(valset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    train_loader = DataLoader(
        trainset,
        batchSize,
        pin_memory=True,
        sampler=sampler_train,
        #shuffle=True,
        num_workers=0,
        collate_fn=None
    )
    val_loader = DataLoader(
        valset,
        batchSize,
        pin_memory=True,
        sampler=sampler_val,
        #shuffle=False,
        num_workers=0,
        collate_fn=None,
    )
    seed = args.seed + Utils_pt.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
   

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module    


    loss = torch.nn.MSELoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=learningRate)

    saveName = f"CLIP-Text-Encoder-{language}"

    best = 50
    best_epoch = 0
    print("Start Training")
    for epoch in tqdm(range(0, epochs)):
        #########
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        #######
        train_stats = train(model, train_loader, device, optim, loss, epoch, tokenizer)

        val_stats = evaluate(model, val_loader, device, tokenizer)
        
    
        
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
        }
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        

        if float(val_stats["mae"]) < best:
            save_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optim.state_dict(),
                "config": config,
                "epoch": epoch,
            }
           
            #hugging face formatTra 
            model.save_pretrained(args.output_dir )
            #pytorch format
            torch.save(
                save_obj,
                os.path.join(args.output_dir, f"checkpoint_best_{language}.pth"),
            )
            

            best = float(val_stats["mae"])
            best_epoch = epoch
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
        
        dist.barrier()  
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/mnt/localdata/karoui/code/Multilingual-CLIP/src/TeacherLearning//m_nlvr2.yaml",
    )
    parser.add_argument("--output_dir", default="output/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    trainStudentTextEncoder(config, args)
