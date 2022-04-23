
'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
####-------------#####
from models.blip_retrieval_multilingual import blip_retrieval_ml
from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()  

    text_ids = []
    text_embeds = []  
    text_atts = []
    image_feats = []
    image_embeds = []
    for text, image, img_id in data_loader: 
            text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
            text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
            text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
            text_embeds.append(text_embed)   
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)
            image = image.to(device) 
            image_feat = model.visual_encoder(image)   
            image_embed = model.vision_proj(image_feat[:,0,:])            
            image_embed = F.normalize(image_embed,dim=-1)      
            
            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)
            
           
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id

    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset),len(data_loader.dataset)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_feats[start+i].repeat(config['k_test'],1,1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[topk_idx], 
                                    attention_mask = text_atts[topk_idx],
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_i2t[start+i,topk_idx] = score + topk_sim
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(data_loader.dataset),len(data_loader.dataset)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
          
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]
    
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result  

@torch.no_grad()
def evaluation_itc(model, data_loader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for itc evaluation...')
    start_time = time.time()  

    
    text_ids = []
    text_embeds = []  
    text_atts = []
    image_feats = []
    image_embeds = []
    for text, image, img_id in data_loader: 
            text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
            text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
            text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
            text_embeds.append(text_embed)   
            text_ids.append(text_input.input_ids)
            text_atts.append(text_input.attention_mask)
            image = image.to(device) 
            image_feat = model.visual_encoder(image)   
            image_embed = model.vision_proj(image_feat[:,0,:])            
            image_embed = F.normalize(image_embed,dim=-1)      
            
            image_feats.append(image_feat.cpu())
            image_embeds.append(image_embed)
            
           
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id

    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)
    
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full((len(data_loader.dataset),len(data_loader.dataset)),-100.0).to(device)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        score_matrix_i2t[start+i,topk_idx] =  topk_sim
        
     
        
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(data_loader.dataset),len(data_loader.dataset)),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        score_matrix_t2i[start+i,topk_idx] = topk_sim
             

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def get_single_embedding(model, text, device, train=False, embed_type=None):
        text = model.tokenizer(text, padding='longest', return_tensors="pt").to(device) 
        text.input_ids[:,0] = model.tokenizer.enc_token_id 
        att = text.attention_mask

        output = model.text_encoder(text.input_ids, 
                                    attention_mask = text.attention_mask,   
                                    return_dict = True,
                                    mode="text"
                                )
        if embed_type == "average":                
            embs = output.last_hidden_state
            sampleLength = att.sum(dim=-1, keepdims=True) 
            maskedEmbs = embs * torch.unsqueeze(att, -1) 
            return maskedEmbs.sum(dim=1) / sampleLength 
        elif embed_type == "cls": 
            return output.last_hidden_state[:,0,:]
        else:
            return output.last_hidden_state

def main(args, config, output_dir):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(f'retrieval_{args.dataset}', config, args)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    _, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
   

    #### Model #### 
    print("Creating model")
     #### Model_german ####
    if args.lan == "de":
        model = blip_retrieval_ml(path = config[f'ml_model_path_{args.lan}_{args.embed_type}'], pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
        model_en = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
        # change weights of special_tokens bos_token:[DEC] and [ENC]: copy them from the original english version of blip
        model.text_encoder.embeddings.word_embeddings.weight[-1, :] = model_en.text_encoder.embeddings.word_embeddings.weight[-1, :] 
        #model.text_encoder.embeddings.word_embeddings.weight[-2, :] = model_en.text_encoder.embeddings.word_embeddings.weight[-2, :]

    else:
        model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    print('Start Evaluation')

    start_time = time.time()    
    if args.itc:             
        score_val_i2t, score_val_t2i, = evaluation_itc(model_without_ddp, val_loader, device, config)  
        score_test_i2t, score_test_t2i = evaluation_itc(model_without_ddp, test_loader, device, config)
    else:
        score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config)  
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)

    if utils.is_main_process():  
    
        val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
        print(val_result)
                              
        test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
        print(test_result)
        
                      
        log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                        **{f'test_{k}': v for k, v in test_result.items()},                  
                    }
        with open(os.path.join(output_dir, "evaluate.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")     
 
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

def get_blip_embeddings(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('retrieval_multi30k', config)  
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,False,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    batch_size=[config['batch_size_test'],config['batch_size_test'],config['batch_size_test']]
    train_loader, val_loader, test_loader = create_loader(datasets,samplers,batch_size=batch_size,
                                                          num_workers=[4,4,4],is_trains=[False,False,False], 
                                                          collate_fns=[None,None,None])
    
    print("train samples", len(datasets[0]))
    #### Model_origiunal #### 
    print("Creating model")
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                                vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                                queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
   

    model = model.to(device) 
   
   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    

    start_time = time.time()

    for text, _, _ in tqdm(train_loader): 
        embs = get_single_embedding(model, text, device, train=False, embed_type=args.embed_type)
        features_np = embs.cpu().data.numpy()
        features_df = pd.DataFrame(features_np)
        if args.embed_type == 'average':
            features_df.to_csv(os.path.join(args.features_dir,'features_train_avg.csv'), mode='a', header=False)
        else:
            features_df.to_csv(os.path.join(args.features_dir,'features_train_cls.csv'), mode='a', header=False)

    print("embeddings saved")

      
        #dist.barrier() 

    

        
        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    parser.add_argument('--exp_name', default=f'test_copy_enc') 
    parser.add_argument('--features_dir', default='/mnt/localdata/karoui/datasets/multi30k/') 
    parser.add_argument('--embed_type', default='avg')   
    parser.add_argument('--lan', default='de') 
    parser.add_argument('--itc', action='store_true')
    parser.add_argument('--dataset', default='multi30k') # or flickr
    parser.add_argument('--config', default='/home/karoui/marvl_project/BLIP/configs/retrieval_multi30k.yaml')  
    
    args = parser.parse_args()
    config_path = f'/home/karoui/marvl_project/BLIP/configs/retrieval_{args.dataset}.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    output_dir = f'output/{args.dataset.upper()}/{args.exp_name}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(output_dir, 'config.yaml'), 'w'))    
    ## to evaluate the model on multi30k data set uncomment the line below
    main(args, config, output_dir)
    #get_blip_embeddings(args, config)
