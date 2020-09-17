from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import re
import glob
import time
import torch
import shutil
import tempfile
import collections
import numpy as np 
import pathlib
import math
_print_freq = 50
_temp_dir = tempfile.mkdtemp()


def set_print_freq(freq):
    global _print_freq
    _print_freq = freq


def get_tmp_dir():
    return _temp_dir


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def create_checkpoint(model, model_ema, optimizer, is_best, is_ema_best,
        best_acc, epoch, root, save_freq=10, prefix='train'):
    pathlib.Path(root).mkdir(parents=True, exist_ok=True) 

    filename = os.path.join(root, '{}_{}.ckpt'.format(prefix, epoch))
    bestname = os.path.join(root, '{}_best.pth'.format(prefix))
    bestemaname = os.path.join(root, '{}_ema_best.pth'.format(prefix))
    tempname = os.path.join(_temp_dir, '{}_tmp.pth'.format(prefix))

    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    if model_ema is not None: 
        if isinstance(model_ema, torch.nn.DataParallel):
            model_ema_state = model_ema.module.state_dict()
        else:
            model_ema_state = model_ema.state_dict()
    else:
        model_ema_state = None

    if is_best:
        torch.save(model_state, bestname)

    if is_ema_best:        
        torch.save(model_ema_state, bestname)
        
    if epoch > 0 and (epoch % save_freq) == 0:
        state = {
            'model': model_state,
            'model_ema': model_ema_state,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }
        torch.save(state, filename)


def resume_checkpoint(model, model_ema, optimizer, root, prefix='train'):
    files = glob.glob(os.path.join(root, "{}_*.ckpt".format(prefix)))

    max_idx = -1
    for file in files:
        num = re.search("{}_(\d+).ckpt".format(prefix), file)
        if num is not None:
            num = num.group(1)
            max_idx = max(max_idx, int(num))

    if max_idx != -1:
        checkpoint = torch.load(
            os.path.join(root, "{}_{}.ckpt".format(prefix, max_idx)))
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])

        if model_ema is not None:
            if isinstance(model_ema, torch.nn.DataParallel):
                model_ema.module.load_state_dict(checkpoint["model_ema"])
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        return (epoch, best_acc)
    else:
        print("==> Can't find checkpoint...training from initial stage")
        return (-1, 0)


def test(val_loader, model, criterion, epoch, train=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.train(train)

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():        
            if not isinstance(model, torch.nn.DataParallel):
                input = input.cuda()
            target = target.cuda(non_blocking=True)        
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)      

        if isinstance(output, tuple):
            loss = criterion(output[0], target_var)
            prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            loss = criterion(output, target_var)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        # record loss and accuracy
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % _print_freq) == 0:
            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i+1, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg


def train_ts(train_loader, model, model_ema, model_t, criterion, optimizer, epoch, metric_map={}, ema_rate=0.9997):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    if model_t is not None:
        model_t.train()

    end = time.time()

    for name, module in model.module.named_modules():      
        if hasattr(module, "_weight_quant"):
            if hasattr(module, "weight_old"):
                del module.weight_old
            module.weight_old = None

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if not isinstance(model, torch.nn.DataParallel):
            input = input.cuda()

        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            if model_t is not None:
                output_t = model_t(input_var)

        # create and attach hook for layer-wise aiwq measure
        hooks = []
        metric_itr_map = {}

        if len(metric_map) > 0:
            def forward_hook(self, input, output):
                if self.weight_old is not None and input[0].get_device() == 0:
                    with torch.no_grad():
                        out_old = torch.nn.functional.conv2d(input[0], self.weight_old, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

                        out_t = torch.transpose(output, 0, 1).contiguous().view(self.out_channels, -1)
                        out_mean = torch.mean(out_t, 1)
                        out_std = torch.std(out_t, 1) # + 1e-8

                        out_old_t = torch.transpose(out_old, 0, 1).contiguous().view(self.out_channels, -1)
                        out_old_mean = torch.mean(out_old_t, 1)
                        out_old_std = torch.std(out_old_t, 1) # + 1e-8

                        out_cond = out_std != 0
                        out_old_cond = out_old_std != 0
                        cond = out_cond & out_old_cond

                        out_mean = out_mean[cond]
                        out_std = out_std[cond]

                        out_old_mean = out_old_mean[cond]
                        out_old_std = out_old_std[cond]

                        KL = torch.log(out_old_std / out_std) + \
                            (out_std ** 2  + (out_mean - out_old_mean) ** 2) / (2 * out_old_std ** 2) - 0.5
                        metric_itr_map[self.name] = KL.mean().data.cpu().numpy()
            
            for name, module in model.module.named_modules():
                if hasattr(module, "_weight_quant") and isinstance(module, torch.nn.Conv2d):
                    module.name = name
                    hooks.append(module.register_forward_hook(forward_hook)) 

        # compute output
        output = model(input_var)
        for hook in hooks:
            hook.remove()
   
        loss_class = criterion(output, target_var)
        if model_t is not None:
            loss_kd = -1 * torch.mean(
                torch.sum(torch.nn.functional.softmax(output_t, dim=1) 
                        * torch.nn.functional.log_softmax(output, dim=1), dim=1))
            loss = loss_class + loss_kd
        else:
            loss = loss_class

        # measure accuracy and record loss
        if isinstance(output, tuple):
            prec1, prec5 = accuracy(output[0].data, target, topk=(1, 5))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss_class.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # store weight for next iteration update
        for name, module in model.module.named_modules():        
            if hasattr(module, "_weight_quant"):
                if name in metric_map:
                    module.weight_old = module._weight_quant().data
                else:
                    if hasattr(module, "weight_old"):
                        del module.weight_old
                    module.weight_old = None

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # implementation of weight exponential moving-average 
        if model_ema is not None:
            for module, module_ema in zip(model.module.modules(), model_ema.module.modules()):
                target = []

                for quant_param in ["a", "b", "c", "d"]:
                    if hasattr(module, quant_param):
                        target.append(quant_param)

                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    target.extend(["weight", "bias"])

                if isinstance(module, (torch.nn.BatchNorm2d)):
                    target.extend(["weight", "bias", "running_mean", "running_var"])

                    if module.num_batches_tracked is not None:
                        module_ema.num_batches_tracked.data = module.num_batches_tracked.data

                for t in target:
                    base = getattr(module, t, None)    
                    ema = getattr(module_ema, t, None)    

                    if base is not None and hasattr(base, "data"):                        
                        ema.data += (1 - ema_rate) * (base.data - ema.data)   

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i+1) % _print_freq) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i+1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
        
        for key, value in metric_itr_map.items():
            if value > 1:
                continue
            metric_map[key] = 0.999 * metric_map[key] + 0.001 * value                
            
    #import operator
    #sorted_x = sorted(metric_map.items(), key=operator.itemgetter(1))
    #print(sorted_x)  


class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """ Implements a schedule where the first few epochs are linear warmup, and
    then there's cosine annealing after that."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_len: int,
                 warmup_start_multiplier: float, max_epochs: int, 
                 eta_min: float = 0.0, last_epoch: int = -1):
        if warmup_len < 0:
            raise ValueError("Warmup can't be less than 0.")
        self.warmup_len = warmup_len
        if not (0.0 <= warmup_start_multiplier <= 1.0):
            raise ValueError(
                "Warmup start multiplier must be within [0.0, 1.0].")
        self.warmup_start_multiplier = warmup_start_multiplier
        if max_epochs < 1 or max_epochs < warmup_len:
            raise ValueError("Max epochs must be longer than warm-up.")
        self.max_epochs = max_epochs
        self.cosine_len = self.max_epochs - self.warmup_len
        self.eta_min = eta_min  # Final LR multiplier of cosine annealing
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.max_epochs:
            raise ValueError(
                "Epoch may not be greater than max_epochs={}.".format(
                    self.max_epochs))
        if self.last_epoch < self.warmup_len or self.cosine_len == 0:
            # We're in warm-up, increase LR linearly. End multiplier is implicit 1.0.
            slope = (1.0 - self.warmup_start_multiplier) / self.warmup_len
            lr_multiplier = self.warmup_start_multiplier + slope * self.last_epoch
        else:
            # We're in the cosine annealing part. Note that the implementation
            # is different from the paper in that there's no additive part and
            # the "low" LR is not limited by eta_min. Instead, eta_min is
            # treated as a multiplier as well. The paper implementation is
            # designed for SGDR.
            cosine_epoch = self.last_epoch - self.warmup_len
            lr_multiplier = self.eta_min + (1.0 - self.eta_min) * (
                1 + math.cos(math.pi * cosine_epoch / self.cosine_len)) / 2
        assert lr_multiplier >= 0.0
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]
