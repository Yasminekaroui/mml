'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_nlvr_multilingual import blip_nlvr

import utils
from utils import cosine_lr_schedule, warmup_lr_schedule
from data import create_dataset, create_sampler, create_loader
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



def copyParams(module_src, module_dest):
    params_src = dict(module_src.state_dict())
    params_dest = dict(module_dest.state_dict())

    corrected_dict = { k.replace("text_encoder.", ''): v for k, v in params_dest.items()}
    for name, param in tqdm(params_src.items()):
        if name in corrected_dict:
            try:
                params_dest["text_encoder."+name].data.copy_(param.data)
            except:
                print("text_encoder."+name)
                
    #new_model = module_dest.load_state_dict(params_dest)            
    return params_dest

def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 10
 
    for i,(image0, image1, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
  
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   

        loss = model(images, text, targets=targets, train=True)    
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())  
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 1
    results = []
  
    for image0, image1, text, targets, question_ids  in metric_logger.log_every(data_loader[0], print_freq, header):
        
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        prediction = model(images, text, targets=targets, train=False)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

        for i in range(pred_class.size(0)):
            results.append(
                {"id": str(question_ids[i].item()),
                "sentence": text,
                "prediction": str(pred_class[i].item()),
                "label": str(targets[i].item())})
     
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, results


def init_experiment(args):
    # init paths of an experiment to save the model and the log file
    exp_name = args.exp_name #+ datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_path = os.path.join(args.output_dir, exp_name)
    if os.path.exists(exp_path):
        pass
    else:
        os.makedirs(exp_path)
    return exp_path 



def main(args, config):
    utils.init_distributed_mode(args) 
    exp_path = init_experiment(args)   
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('marvl', config) 
    
    
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,False,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    batch_size=[config['batch_size_train'],config['batch_size_test'],config['batch_size_test']]
    #train_loader, val_loader, test_loader = create_loader(datasets,samplers,batch_size=batch_size,
                                                          #num_workers=[4,4,4],is_trains=[True,False,False], 
                                                          #collate_fns=[None,None,None])
    ##new
    language_id = {'id':0, 'sw':1,'ta':2,'tr':3,'zh':4}
    dataset = datasets[language_id[args.lan]]
    test_loader = create_loader([dataset],[samplers[2]],batch_size=[config['batch_size_test']],
                                                          num_workers=[4],is_trains=[False], 
                                                          collate_fns=[None])
    
    #### Model #### 
    print("Creating model")
    model = blip_nlvr(path = config[f'ml_model_path_{args.lan}_{args.embed_type}'],pretrained=config['pretrained'], image_size=config['image_size'], 
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])


    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
            
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    print("Start Testing")
    start_time = time.time()



      
    test_stats, results = evaluate(model, test_loader, device, config)  # I added results
    
    if utils.is_main_process():
        log_stats = {**{f'{args.lan}test_{k}': v for k, v in test_stats.items()},
                        }
        with open(os.path.join(exp_path, f"log_{args.lan}.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        json.dump(results, open(os.path.join(exp_path, f"ml_results_{args.lan}.json"), "w"))
           
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/marvl.yaml')
    parser.add_argument('--output_dir', default='output/MARVL/tr_cls')
    parser.add_argument('--exp_name', default='average_new_real')
    parser.add_argument('--evaluate', action='store_true') 
    ##new
    parser.add_argument('--lan', default='tr')
    parser.add_argument('--embed_type', default='cls')#or avg
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)