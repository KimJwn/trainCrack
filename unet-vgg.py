import argparse
import os
from re import M
import csv
from pyrsistent import b
import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

#from sre_parse import _OpGroupRefExistsType
# from unittest.mock import patch

# Custome Dataset for crack data
from data.Dataset import *

# Metric and logging
from utils.train_utils import *
from utils.metrics import *

# convolution models for semantic segmentation
from model.unet import UNet16
# from model import hardnet as Hard

# Weights & Biases
import wandb

save_dir = '/home/jovyan/DragonBall/'



def parse_args():
    parser = argparse.ArgumentParser(description='CRACK SEGMENTATION')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--train', '-t', default=True)
    parser.add_argument('--project_name', '-n', default='crack_seg')
    parser.add_argument('--no_cuda', '-c', default=False)
    parser.add_argument('--model', '-m', default='Unet')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epoch', '-e', type=int, default=15)
    parser.add_argument('--lr', '-l', default=0.001, type=float)
    # parser.add_argument('—save_model', '-s', action='store_true', default=False) lr

    return parser.parse_args()


def train(train_loader, model, optimizer, criterion, logger, device, epoch, batch_size):

    model.train()
    print("start train {}".format(len(train_loader)))

    metrics = []
    losses = []


    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description(f'\nTrain Epoch {epoch}')
    
    for batch_idx, (image, label) in enumerate(train_loader):
        with torch.autograd.detect_anomaly():
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
        metric = torch_iou_fast(outputs, label)
        metrics.append(metric)
        # 학습 상황 출력
        if batch_idx % 10 == 0:
            if int(os.environ["LOCAL_RANK"]) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'
                    .format(epoch, batch_idx * len(label), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader),
                                loss.item(), metric))
    #log
    log = {
        "train_metrics":  (sum(metrics)/len(metrics)), 
        "tarin_losses" : (sum(losses)/len(losses)) 
    }
    logger.info(log)
    wandb.log(log) 
    return log


def test(test_loader, model, criterion, logger, device, epoch, batch_size):
    model.eval()

    metrics = []
    losses = []

    tq = tqdm.tqdm(total=(len(test_loader) * batch_size))
    # tq.set_description(f'Test Epoch {epoch}')
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            # print(batch_idx)
            # image, label = image.to(torch.), label.to(device)
            
            # Forward 
            outputs = model(image) 
            # loss_func 
            loss = criterion(outputs, label)
            losses.append(loss)

            # Log training loss, metrics

            metric = torch_iou_fast(outputs, label)
            metrics.append(metric)

            # 학습 상황 출력
            if batch_idx % 10 == 0:
                if int(os.environ["LOCAL_RANK"]) == 0:
                    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'
                        .format(epoch, batch_idx * len(label), len(test_loader.dataset),
                                    100. * batch_idx / len(test_loader),
                                    loss.item(), metric))                     
    #log
    log = {
        "test_metrics":  (sum(metrics)/len(losses)),#/ (batch_size*epoches),
        "test_losses" :  (sum(losses)/len(losses))#/ (batch_size*epoches)
    }
    logger.info(log)
    wandb.log(log) 
    return log



def main():
    wandb.login()
    device_id = int(os.environ["LOCAL_RANK"])
    # print(torch.cuda.nccl.version())
    # dist.init_process_group(backend='nccl', 
    #                         init_method='tcp://0.0.0.0:29500',
    #                         world_size=2,
    #                         rank = device_id)

    args = parse_args()
    wandb.init(project="Unet", entity="kau-aiclops", name="G1")
    wandb.config.update(args) # adds all of the arguments as config variables
    torch.manual_seed(args.seed)

    #
    lr = args.lr
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 4
    epoches = args.epoch # 15
    

    if device_id == 0:
        print(torch.__version__, torch.cuda.__path__)	
        print("gpu") if torch.cuda.is_available() else print("cpu")
        print("dist") if dist.is_available() else print("can not use dist")
        # use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 4


    #
    path = create_save_dir(args.project_name)
    logger = get_log(path)


    #
    train_set = CrackDataset(split='train')
    test_set = CrackDataset(split='test')

    train_loader = get_loader(train_set, batch_size=args.batch_size, num_workers=num_workers, mode='mini')
    test_loader = get_loader(test_set, batch_size=args.batch_size, num_workers=num_workers, mode="mini")
      
    #
    model = UNet16(pretrained=False)
    # model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])
    wandb.watch(model)

    #
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    #
    criterion = torch.nn.BCEWithLogitsLoss().to('cuda')
    batch_size=args.batch_size

    for epoch in range(epoches):
        # train
        train_log = train(train_loader, model, optimizer, criterion, logger, device, epoch, batch_size)
        # dist.all_reduce
        #test
        test_log = test(test_loader, model, criterion, logger, device, epoch, batch_size)
        #log
        log = {"epoch": epoch, "train": train_log, "test": test_log}
        logger.info(log)
        wandb.log(log) 


if __name__ == "__main__":
    main()


"""
def parse_args():
    parser = argparse.ArgumentParser(description='CRACK SEGMENTATION')
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--train', '-t', default=True)
    parser.add_argument('--model', '-m', default="Unet")
    parser.add_argument('--project_name', '-n', default='crack_seg')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('—save_model', '-s', action='store_true', default=False)

    return parser.parse_args()



def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def train(train_loader, model, optimizer, criterion, logger, device):
    model.train()

    metrics = []
    losses = []
    
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        # Forward 
        outputs = model(image)
        # loss_func
        loss = criterion(outputs, label)

        # Log training loss, metrics
        metric = torch_iou(outputs, label)
        metrics.append(metric)
        losses.append(loss)

        # Gradinet 
        loss.backward()   
        # weight update 
        optimizer.step() 
    #
    return metrics, losses

'''
def validation(model, 
               val_loader, 
               logger, ):
    model.eval()
'''

def test(test_loader, model, criterion, logger, device):
    model.eval()

    metrics = []
    losses = []

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            # Forward 
            outputs = model(image) 
            # loss_func 
            loss = criterion(outputs, label)

            # Log training loss, metrics
            metric = torch_iou(outputs, label)
            metrics.append(metric)
            losses.append(loss)

        wandb.log({
        "Test Accuracy": 100. * (sum(metrics) / len(test_loader.dataset)),
        "Test Loss": (sum(metrics) / len(test_loader.dataset)) })

    return metrics, losses


def main():
    #wandb.login()

    args = parse_args()
    wandb.init(project=args.proj_name, entity="AIclops", name=args.model)
    wandb.config.update(args) # adds all of the arguments as config variables

    model = args.model
    print(model)

    #
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    batch_size = 4
    num_workers = 4
    epoches = 10 # 50
    batch_size, num_workers = 16,4
    
    torch.manual_seed(args.seed)

    #
    print(torch.__version__, torch.cuda.__path__)	
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device_id = int(os.environ["LOCAL_RANK"])

    path = create_save_dir(args.pr
    oject_name)

    logger = get_log(path)
 
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("gpu") if torch.cuda.is_available() else print("cpu")

    #
    train_set = CrackDataset(split='train')
    test_set = CrackDataset(split='test')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=num_workers)
    test_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=num_workers)
      


    #
    if args.model == "Unet":
        model = UNet16(pretrained=False)
        wandb.watch(model)
    # elif args.model == "Hard":
    #     model = Hard()
    #     wandb.watch(model)
    else :
        print("Error! : model unvalid select, ", model)
    


    #
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    #
    criterion = torch.nn.BCEWithLogitsLoss().to('cuda')

    if args.save_model:
        model_fname = "{}{}{}.pth".format(args.project_name, args.model, epoches)
        torch.save(model.state_dict(), model_fname)

    for epoch in range(epoches):
        # train
        metrics, losses = train(train_loader, model, optimizer, criterion, logger, device)  # type: ignore

        #test
        test_metrics, test_losses = test(test_loader, model, criterion, logger, device)  # type: ignore
    
        #log
        log = {
            "epoch": epoch,
            "train_metrics" : metrics,
            "train_loss" : losses,
            "test_metrics": test_metrics,
            "test_losses" : test_losses
        }
        
        with open(path+'log_csv.csv','a') as f:
            w = csv.writer(f)
            if epoch == 0:
                w.writerow(log.keys())
                w.writerow(log.values())
            else:
                w.writerow(log.values())
    
if __name__ == "__main__":
    main()
    """