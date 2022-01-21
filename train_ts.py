''' PROFIT Implementation '''
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import datetime
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import argparse
from my_lib.train_test import (
    resume_checkpoint,
    create_checkpoint,
    test,
    train_ts,
    CosineWithWarmup
)
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/SSD/ILSVRC2012")
parser.add_argument("--ckpt", required=True, help="checkpoint directory")

parser.add_argument("--quant_op", choices=["duq"])
parser.add_argument("--model", choices=["mobilenetv2", "mobilenetv3"])
parser.add_argument("--teacher", choices=["none", "self", "resnet101"])

parser.add_argument("--lr", default=0.04, type=float)
parser.add_argument("--decay", default=2e-5, type=float)

parser.add_argument("--warmup", default=3, type=int)
parser.add_argument("--bn_epoch", default=5, type=int)
parser.add_argument("--ft_epoch", default=15, type=int)
parser.add_argument("--sample_epoch", default=5, type=int)

parser.add_argument("--use_ema", action="store_true", default=False)
parser.add_argument("--stabilize", action="store_true", default=False)

parser.add_argument("--w_bit", required=True, type=int, nargs="+")
parser.add_argument("--a_bit", required=True, type=int, nargs="+")
parser.add_argument("--w_profit", required=True, type=int, nargs="+")

args = parser.parse_args()
ckpt_root = args.ckpt   # "/home/eunhyeokpark/cifar10/"
data_root = args.data  # "/home/eunhyeokpark/cifar10/"
use_cuda = torch.cuda.is_available()

print("==> Prepare data..")
from my_lib.imagenet import get_loader
testloader, trainloader, _ = get_loader(data_root, test_batch=256, train_batch=256,)

if args.quant_op == "duq":
    from quant_op.duq import QuantOps
    print("==> differentiable and unified quantization method is selected..")
else:
    raise NotImplementedError

print("==> Student model: %s" % args.model)
if args.model == "mobilenetv2":
    from conv_model.ilsvrc.MobileNetV2_quant import mobilenet_v2
    model = mobilenet_v2(QuantOps)
    model.load_state_dict(torch.load("./pretrained/mobilenet_v2-b0353104.pth"), False)
elif args.model == "mobilenetv3":
    from conv_model.ilsvrc.MobileNetV3Large_pad_quant import MobileNetV3Large
    model = MobileNetV3Large(QuantOps)
    model.load_state_dict(torch.load("./pretrained/mobilenet_v3_pad.pth"), False)
else:
    raise NotImplementedError

print("==> Teacher model: %s" % args.teacher)
if args.teacher == "none":
    model_t = None
elif args.teacher == "self":    
    model_t = copy.deepcopy(model)
elif args.teacher == "resnet101":
    from torchvision.models.resnet import resnet101
    model_t = resnet101(True)
else:
    raise NotImplementedError

if  model_t is not None:
    for params in model_t.parameters():
        params.requires_grad = False

model_ema = None
if args.use_ema:
    model_ema = copy.deepcopy(model)

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    if model_t is not None:
        model_t.cuda()
        model_t = torch.nn.DataParallel(model_t, device_ids=range(torch.cuda.device_count()))

    if model_ema is not None:
        model_ema.cuda()
        model_ema = torch.nn.DataParallel(model_ema, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


def categorize_param(model, skip_list=()):
    quant = []
    skip = []
    bnbias = []
    weight = []

    for name, param in model.named_parameters():  
        skip_found = False
        for s in skip_list:
            if name.find(s) != -1:
                skip_found = True

        if not param.requires_grad:
            continue  # frozen weights    
        elif name.endswith(".a") or name.endswith(".b") \
            or name.endswith(".c") or name.endswith(".d"):
            quant.append(param)
        elif skip_found:
            skip.append(param)        
        elif len(param.shape) == 1 or name.endswith(".bias"):
            bnbias.append(param)
        else:
            weight.append(param)

    return (quant, skip, weight, bnbias)


def get_optimizer(params, train_quant, train_weight, train_bnbias):
    (quant, skip, weight, bnbias) = params
    optimizer = optim.SGD([
        {'params': skip, 'weight_decay': 0, 'lr': 0},
        {'params': quant, 'weight_decay': 0., 'lr': args.lr * 1e-2 if train_quant else 0},
        {'params': bnbias, 'weight_decay': 0., 'lr': args.lr if train_bnbias else 0},
        {'params': weight, 'weight_decay': args.decay, 'lr': args.lr if train_weight else 0},
    ], momentum=0.9, nesterov=True)
    return optimizer


def phase_prefix(a_bit, w_bit):
    prefix_base = "ts_%s_%s_%s_" % (args.model, args.teacher, args.quant_op)
    return prefix_base + ("ema_" if args.use_ema else "") + ("%d_%d" % (a_bit, w_bit))


def train_epochs(optimizer, warmup_len, max_epochs, prefix):
    last_epoch, best_acc = resume_checkpoint(model, model_ema, optimizer, args.ckpt, prefix)

    scheduler = CosineWithWarmup(optimizer, 
                        warmup_len=warmup_len, warmup_start_multiplier=0.1,
                        max_epochs=max_epochs, eta_min=1e-3, last_epoch=last_epoch)
                        
    for epoch in range(last_epoch+1, max_epochs):
        train_ts(trainloader, model, model_ema, model_t, criterion, optimizer, epoch)
        acc_base = test(testloader, model, criterion, epoch)

        acc_ema = 0        
        if model_ema is not None:
            acc_ema = test(testloader, model_ema, criterion, epoch)

        is_best = False
        if acc_base > best_acc:
            is_best = True
        
        is_ema_best = False
        if acc_ema > best_acc:
            is_ema_best =  True
        
        best_acc = max(best_acc, acc_base, acc_ema)        
        create_checkpoint(model, model_ema, optimizer,
                          is_best, is_ema_best, best_acc, epoch, ckpt_root, 1, prefix)    
        scheduler.step()  
    return best_acc


# full-precision teacher-student boosting
a_bit, w_bit = 32, 32

if args.teacher != "none": # full-precision fine-tuning with teacher-student
    prefix = phase_prefix(a_bit, w_bit)

    print("==> Full precision fine-tuning")
    params = categorize_param(model)
    optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True)    
    best_acc = train_epochs(optimizer, args.warmup, args.ft_epoch, prefix)


# progressive activation quantization 
QuantOps.initialize(model, trainloader, 2**args.a_bit[0], act=True)
if model_ema is not None:
    QuantOps.initialize(model_ema, trainloader, 2**args.a_bit[0], act=True)

for a_bit in args.a_bit:
    prefix = phase_prefix(a_bit, w_bit)
    print("==> Activation quantization, bit %d" % a_bit)
    
    for name, module in model.named_modules():
        if isinstance(module, (QuantOps.ReLU, QuantOps.ReLU6, QuantOps.Sym, QuantOps.HSwish)):
            module.n_lv = 2 ** a_bit

    if model_ema is not None:
        for name, module in model_ema.named_modules():
            if isinstance(module, (QuantOps.ReLU, QuantOps.ReLU6, QuantOps.Sym, QuantOps.HSwish)):
                module.n_lv = 2 ** a_bit

    if args.stabilize:
        print("==> BN stabilize")
        params = categorize_param(model)
        optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True) 
        best_acc = train_epochs(optimizer, 0, args.bn_epoch, prefix + "_bn")

    print("==> Fine-tuning")
    optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True) 
    best_acc = train_epochs(optimizer, args.warmup, args.ft_epoch, prefix)

    if args.stabilize:
        print("==> BN stabilize 2")
        optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True) 
        best_acc = train_epochs(optimizer, 0, args.bn_epoch, prefix + "_bn2")


# progressive weight quantization
with torch.no_grad():
    QuantOps.initialize(model, trainloader, 2**args.w_bit[0], weight=True)

    if model_ema is not None:
        QuantOps.initialize(model_ema, trainloader, 2**args.w_bit[0], weight=True)

for w_bit in args.w_bit:
    prefix = phase_prefix(a_bit, w_bit)
    print("==> Weight quantization, bit %d" % w_bit)
    
    for name, module in model.named_modules():
        if isinstance(module, (QuantOps.Conv2d, QuantOps.Conv2dPad, QuantOps.Linear)):
            module.n_lv = 2 ** w_bit

    if model_ema is not None:
        for name, module in model_ema.named_modules():
            if isinstance(module, (QuantOps.Conv2d, QuantOps.Conv2dPad, QuantOps.Linear)):
                module.n_lv = 2 ** w_bit

    if args.stabilize:
        print("==> BN stabilize")
        params = categorize_param(model)
        optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True) 
        best_acc = train_epochs(optimizer, 0, args.bn_epoch, prefix + "_bn")


    if w_bit in args.w_profit:    # PROFIT training
        print("==> Sampling")
        metric_map = {}
        for name, module in model.module.named_modules():
            if hasattr(module, "_weight_quant") and isinstance(module, nn.Conv2d):
                metric_map[name] = 0

        #if os.path.exists(os.path.join(args.ckpt, prefix + ".pkl")):
        if os.path.exists(os.path.join(args.ckpt, prefix + ".pkl")) and False:
            print("==> Load existed sampled map")
            with open(os.path.join(args.ckpt, prefix + ".pkl"), "rb") as f:
                metric_map = pickle.load(f)

        else:
            optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True) 
            
            for epoch in range(args.sample_epoch):        
                train_ts(trainloader, model, model_ema, model_t, criterion, optimizer, epoch, metric_map)  
                acc = test(testloader, model, criterion, epoch)

                if model_ema is not None:
                    acc = test(testloader, model_ema, criterion, epoch)

            with open(os.path.join(args.ckpt, prefix + ".pkl"), "wb") as f:
                pickle.dump(metric_map, f)  
        
        skip_list = []
        import operator
        sort = sorted(metric_map.items(), key=operator.itemgetter(1), reverse=True)
        for s in sort[0:int(len(sort) * 1/3)]:
            skip_list.append(s[0])

        skip_list_next = []
        for s in sort[int(len(sort) * 1/3):int(len(sort) * 2/3)]:
            skip_list_next.append(s[0])

        params = categorize_param(model)
        optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True) 
        best_acc = train_epochs(optimizer, args.warmup, args.ft_epoch, prefix + "_ft1")

        params = categorize_param(model, skip_list)
        optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True) 
        best_acc = train_epochs(optimizer, args.warmup, args.ft_epoch, prefix + "_ft2")

        params = categorize_param(model, skip_list + skip_list_next)
        optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True) 
        best_acc = train_epochs(optimizer, args.warmup, args.ft_epoch, prefix + "_ft3")

    else:                     
        print("==> Fine-tuning")
        params = categorize_param(model)
        optimizer = get_optimizer(params, train_quant=True, train_weight=True, train_bnbias=True) 
        best_acc = train_epochs(optimizer, args.warmup, args.ft_epoch, prefix)

    if args.stabilize:
        print("==> BN stabilize 2")
        optimizer = get_optimizer(params, train_quant=True, train_weight=False, train_bnbias=True) 
        best_acc = train_epochs(optimizer, 0, args.bn_epoch, prefix + "_bn2")

print("==> Finish training.. best accuracy is {}".format(best_acc))
