# JW 
#   Dragonball
#       main.py

#torchrun --nproc_per_node 1 /home/jovyan/G1/dragon_ball/distribute_ex.py -b 16 -e 20 -n Unet-vgg-3 -l 0.05

#torchrun --nproc_per_node 1 /home/jovyan/DragonBall/distribute_ex.py -b 4 -e 2 -n Unet_vgg_ujin -l 0.01

import argparse
import os
from re import M
import csv
from pyrsistent import b
import tqdm

import torch
import torch.distributed as dist

# Custome Dataset for crack data
from data.Dataset import *

# Metric and logging
from utils.train_utils import *
from utils.metrics import *

# convolution models for semantic segmentation
from model.unet import *

from torch.nn.parallel import DistributedDataParallel as DDP
# Weights & Biases
import wandb


save_dir = '/home/jovyan/G1/dragon_ball/utils'

def parse_args():
    parser = argparse.ArgumentParser(description='CRACK SEGMENTATION')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--train', '-t', default=True)
    parser.add_argument('--project_name', '-n', default='crack_seg')
    parser.add_argument('--data_loader', '-d', default='mini')
    parser.add_argument('--no_cuda', '-c', default=False)
    parser.add_argument('--model', '-m', default='Unet')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epoch', '-e', type=int, default=15)
    parser.add_argument('--lr', '-l', default=0.001, type=float)
    # parser.add_argument('—save_model', '-s', action='store_true', default=False) lr

    return parser.parse_args()


def train(train_loader, model, optimizer, criterion, device, epoch, batch_size):
    model.train()
    if epoch == 0: print("start train {}".format(len(train_loader)))

    metrics = []
    losses = []

    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description(f'Train Epoch {epoch}')
    
    for batch_idx, (image, label) in enumerate(train_loader):
        # with torch.autograd.detect_anomaly():
        # image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        # Forward 
        outputs = model(image)
        # loss_func
        loss = criterion(outputs, label)
        losses.append(loss.item())
        # Gradinet 

        loss.backward()   
        # weight update 
        optimizer.step() 

        # loss = loss.to(torch.device)
        # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        # dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        # loss = loss/2
        # metric = metric/2

        # Log training loss, metrics
        # print(outputs.shape)
        # metric = uj_dice(outputs, label)
        # metric_f = torch_iou_fast(outputs, label)
        print(outputs.shape)
        m_j = batch_iou(outputs, label)
        metrics.append(m_j)

        tq.set_postfix(loss='{:.5f}'.format((sum(losses)/len(losses))), jcAcc ='{:.5f}'.format(sum(metrics)/len(metrics)))
        tq.update(batch_size)
        # # 학습 상황 출력
        # if int(os.environ["LOCAL_RANK"]) == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc uj dc: {:.6f}\tAcc fs: {:.6f}\t*Acc jc: {:.6f}*'
        #         .format(epoch, batch_idx * len(label), len(train_loader.dataset),
        #                     100. * batch_idx / len(train_loader),
        #                     loss.item(), metric, metric_f, m_j))
    tq.close()
    return (sum(metrics)/len(metrics)), (sum(losses)/len(losses)) 


def test(test_loader, model, criterion, device, epoch, batch_size):
    model.eval()

    metrics = []
    losses = []

    tq = tqdm.tqdm(total=(len(test_loader) * batch_size))
    tq.set_description(f'Test Epoch {epoch}')
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            # print(batch_idx)
            # image, label = image.to(torch.), label.to(device)
            
            # Forward 
            outputs = model(image) 
            # loss_func 
            loss = criterion(outputs, label)
            losses.append(loss.item())

            # Log training loss, metrics

            # metric = uj_dice(outputs, label)
            # metric_f = torch_iou_fast(outputs, label)
            print(outputs.shape)
            m_j = get_iou_vector(outputs, label)
            metrics.append(m_j)

            tq.set_postfix(loss='{:.5f}'.format((sum(losses)/len(losses))), jcAcc ='{:.5f}'.format(sum(metrics)/len(metrics)))
            tq.update(batch_size)

            # 학습 상황 출력
            # if batch_idx % 10 == 0:
            # if int(os.environ["LOCAL_RANK"]) == 0:
                # print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc uj dc: {:.6f}\tAcc fs: {:.6f}\tAcc jc: {:.6f}'
                #     .format(epoch, batch_idx * len(label), len(test_loader.dataset),
                #                 100. * batch_idx / len(test_loader),
                #                 loss.item(), metric, metric_f, m_j))                     
    tq.close()
    return (sum(metrics)/len(losses)), (sum(losses)/len(losses))



def main():
    args = parse_args()   
    torch.manual_seed(args.seed)
    device_id = int(os.environ["LOCAL_RANK"])
    # print(torch.cuda.nccl.version())
    # dist.init_process_group(backend='nccl', 
    #                         init_method='tcp://0.0.0.0:29500',
    #                         world_size=2,
    #                         rank = device_id)

    #
    wandb.login()
    wandb.init(project="Unet", entity="kau-aiclops", name=args.project_name)
    wandb.config.update(args) # adds all of the arguments as config variables

    #
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 4
    epoches = args.epoch # 15

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_id == 0:
        print(torch.__version__, torch.cuda.__path__, device)	
        print("dist") if dist.is_available() else print("can not use dist")
        # use_cuda = not args.no_cuda and torch.cuda.is_available()
    


    #
    path = create_save_dir(args.project_name)
    logger = get_log(path)

    #
    train_set = CrackDataset(split='train')
    test_set = CrackDataset(split='test')

    train_loader = get_loader(train_set, batch_size=args.batch_size, num_workers=num_workers, mode=args.data_loader)
    test_loader = get_loader(test_set, batch_size=args.batch_size, num_workers=num_workers, mode=args.data_loader)
      
    #
    model = UNet16(pretrained=False)
    # model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])
    wandb.watch(model)

    #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss().to('cuda')
    batch_size=args.batch_size

    for epoch in range(epoches):
        print('\n')

        # train
        trainM, trainL = train(train_loader, model, optimizer, criterion, device, epoch, batch_size)

        # dist.all_reduce
        #test
        testM, testL = test(test_loader, model, criterion, device, epoch, batch_size)

        #log
        log = {"epoch": epoch, "Train Loss":trainL,  "Train Acc":trainM,  "Test Loss":testL,  "Test Acc":testM}
        logger.info(log)
        wandb.log(log) 

        with open(path+'log_csv.csv','a') as f:
            w = csv.writer(f)
            if epoch == 0:
                w.writerow(log.keys())
                w.writerow(log.values())
            else:
                w.writerow(log.values())
    

if __name__ == "__main__":
    main()