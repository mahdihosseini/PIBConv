import os
import sys
import time
import glob
import utils
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from copy import deepcopy
from numpy import linalg as LA
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from adas import Adas
from adas.metrics import Metrics

# for ADP dataset
from ADP_utils.classesADP import classesADP


parser = argparse.ArgumentParser("adaptive_darts")
#################### 
# Dataset
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='ADP-Release1', help='valid datasets: cifar10, cifar100, ADP-Release1')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--image_size', type=int, default=64, help='CPATH image size')
# color augmentation
parser.add_argument('--color_aug', action='store_true', default=False, help='use color augmentation')
parser.add_argument('--color_distortion', type=float, default=0.3, help='color distortion param')
# For ADP dataset only
parser.add_argument('--adp_level', type=str, default='L3', help='ADP level')
#################### 
# Training details
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.175, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding') 
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--learnable_bn', action='store_true', default=False, help='learnable parameters in batch normalization')
# Gumbel-softmax
parser.add_argument('--gumbel', action='store_true', default=False, help='use or not Gumbel-softmax trick')
parser.add_argument('--tau_max', type=float, default=10.0, help='initial tau')
parser.add_argument('--tau_min', type=float, default=1.0, help='minimum tau')
# Adas optimizer
parser.add_argument('--adas', action='store_true', default=False, help='whether or not to use adas optimizer')
parser.add_argument('--scheduler_beta', type=float, default=0.98, help='beta for lr scheduler')
parser.add_argument('--scheduler_p', type=int, default=1, help='p for lr scheduler')
parser.add_argument('--step_size', type=int, default=50, help='step_size for dropping lr')
parser.add_argument('--gamma', type=float, default=1.0, help='gamma for dropping lr')
####################
# Model details
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=4, help='total number of layers')
parser.add_argument('--node', type=int, default=4, help='number of nodes in a cell')
####################
# Others
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--file_name', type=str, default='_', help='metrics and weights data file name')

args = parser.parse_args()

args.save = 'Search-{}-data-{}-{}'.format(args.save, args.dataset, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'cifar100':
    n_classes = 100
    data_folder = 'cifar-100-python'
elif args.dataset == 'cifar10':
    n_classes = 10
    data_folder = 'cifar-10-batches-py'
elif args.dataset == 'ADP-Release1':
    n_classes = classesADP[args.adp_level]['numClasses']
else:
    logging.info('dataset not supported')
    sys.exit(1)

is_multi_gpu = False

def main():
    global is_multi_gpu

    gpus = [int(i) for i in args.gpu.split(',')]
    logging.info('gpus = %s' % gpus)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if len(gpus) == 1:
        torch.cuda.set_device(int(args.gpu))
    else:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        is_multi_gpu = True
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    # load dataset
    if args.dataset == 'cifar100':
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar10':
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'ADP-Release1':
        train_transform, valid_transform = utils._data_transforms_adp(args)
        train_data = utils.ADP_dataset(level=args.adp_level, transform=train_transform, root=args.data, split='train_search', portion=args.train_portion)
        valid_data = utils.ADP_dataset(level=args.adp_level, transform=train_transform, root=args.data, split='valid_search', portion=args.train_portion)

    if args.dataset in ['cifar100', 'cifar10']:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0)

        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=0)
    elif args.dataset == 'ADP-Release1':
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.RandomSampler(train_data),
            pin_memory=True, num_workers=0)

        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.RandomSampler(valid_data),
            pin_memory=True, num_workers=0)

    # build network
    if args.dataset in ['cifar100', 'cifar10']:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
    elif args.dataset == 'ADP-Release1':
        dataset_size = len(train_queue.dataset)
        print('train dataset size:', len(train_queue.dataset))
        print('valid dataset size:', len(valid_queue.dataset))

        train_class_counts = np.sum(train_queue.dataset.class_labels, axis=0)
        weightsBCE = dataset_size / train_class_counts
        weightsBCE = torch.as_tensor(weightsBCE, dtype=torch.float32).to(int(args.gpu))
        criterion = torch.nn.MultiLabelSoftMarginLoss(weight=weightsBCE).cuda()

    model = Network(args.init_channels, n_classes, args.layers, criterion, learnable_bn=args.learnable_bn, steps=args.node, multiplier=args.node)
    if is_multi_gpu:
        model = nn.DataParallel(model)
    model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    arch_parameters = model.module.arch_parameters() if is_multi_gpu else model.arch_parameters()
    arch_params = list(map(id, arch_parameters))
    model_parameters = model.module.parameters() if is_multi_gpu else model.parameters()
    model_params = filter(lambda p: id(p) not in arch_params, model_parameters)
    
    # Optimizer for model weights update
    # Use Adas: optimizer and scheduler
    if args.adas:
        optimizer = Adas(params=list(model_params),
            lr=args.learning_rate,
            beta=args.scheduler_beta,
            step_size=args.step_size,
            gamma=args.gamma,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    # Use SGD: default in DARTS paper
    else:
        optimizer = torch.optim.SGD(
            model_params,
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, criterion, args)

    if not args.adas:
        # record probing metrics
        arch_parameters = model.module.arch_parameters() if is_multi_gpu else model.arch_parameters()
        arch_params = list(map(id, arch_parameters))
        model_parameters = model.module.parameters() if is_multi_gpu else model.parameters()
        model_params = filter(lambda p: id(p) not in arch_params, model_parameters)

        metrics = Metrics(params=list(model_params))

    # files to record searching results
    performance_statistics = {}
    arch_statistics = {}
    genotype_statistics = {}
    metrics_path = '../save_data/metrics_stat_{}.xlsx'.format(args.file_name)
    weights_path = '../save_data/weights_stat_{}.xlsx'.format(args.file_name)
    genotypes_path = '../save_data/genotypes_stat_{}.xlsx'.format(args.file_name)

    errors_dict = {'train_acc_1': [], 'train_loss': [], 'valid_acc_1': [], 'valid_loss': []}

    for epoch in range(args.epochs):
        if args.adas:
            lr = optimizer.lr_vector
        else:
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.module.genotype() if is_multi_gpu else model.genotype()
        logging.info('epoch: %d', epoch)
        logging.info('genotype = %s', genotype)

        # training
        train_acc_1, train_acc_5, train_obj = train(epoch, train_queue, valid_queue,
                                     model, architect, criterion,
                                     optimizer, lr)
        print('\n')
        logging.info('train_acc_1 %f, train_acc_5 %f', train_acc_1, train_acc_5)

        # validation
        valid_acc_1, valid_acc_5, valid_obj = infer(valid_queue, model, criterion)
        print('\n')
        logging.info('valid_acc_1 %f, valid_acc_5 %f', valid_acc_1, valid_acc_5)

        # update the errors dictionary
        errors_dict['train_acc_1'].append(train_acc_1)
        errors_dict['train_loss'].append(train_obj)
        errors_dict['valid_acc_1'].append(valid_acc_1)
        errors_dict['valid_loss'].append(valid_obj)

        # update network metrics (knowledge gain, condition mapping, etc)
        if args.adas:
            # AdaS: update learning rates
            optimizer.epoch_step(epoch)
            io_metrics = optimizer.KG
            lr_metrics = optimizer.velocity
        else:
            metrics()
            io_metrics = metrics.KG(epoch)
            lr_metrics = None
        # weights
        weights_normal = F.softmax(model.module.alphas_normal if is_multi_gpu else model.alphas_normal, dim=-1).detach().cpu().numpy()
        weights_reduce = F.softmax(model.module.alphas_reduce if is_multi_gpu else model.alphas_reduce, dim=-1).detach().cpu().numpy()

        # write data to excel files
        write_data(epoch, io_metrics, lr_metrics, weights_normal, weights_reduce, genotype,
                   performance_statistics, arch_statistics, genotype_statistics,
                   metrics_path, weights_path, genotypes_path)

        # save model parameters
        save_model = model.module if is_multi_gpu else model
        utils.save(save_model, os.path.join(args.save, 'weights.pt'))

def train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    global is_multi_gpu
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    trained_data_size = 0
    for step, (input, target) in enumerate(train_queue):
        # one mini-batch
        print('\rtrain mini batch {:03d}'.format(step), end=' ')
        model.train()
        n = input.size(0)
        trained_data_size += n

        if args.gumbel:
            model.module.set_tau(args.tau_max - epoch * 1.0 / args.epochs * (args.tau_max - args.tau_min)) if is_multi_gpu \
            else model.set_tau(args.tau_max - epoch * 1.0 / args.epochs * (args.tau_max - args.tau_min))

        input = input.cuda()
        target = target.cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda()

        # logging.info('update arch...')
        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        # logging.info('update weights...')
        optimizer.zero_grad()

        logits = model(input, gumbel=args.gumbel)
        loss = criterion(logits, target)

        loss.backward()
        arch_parameters = model.module.arch_parameters() if is_multi_gpu else model.arch_parameters()
        arch_params = list(map(id, arch_parameters))

        model_parameters = model.module.parameters() if is_multi_gpu else model.parameters()
        model_params = filter(lambda p: id(p) not in arch_params, model_parameters)
        nn.utils.clip_grad_norm_(model_params, args.grad_clip)
        optimizer.step()

        if args.dataset in ['cifar100', 'cifar10']:
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
        elif args.dataset == 'ADP-Release1':
            m = nn.Sigmoid()
            preds = (m(logits) > 0.5).int()
            prec1, prec5 = utils.accuracyADP(preds, target)
            objs.update(loss.item(), n)
            top1.update(prec1.double(), n)
            top5.update(prec5.double(), n)

        if step % args.report_freq == 0:
            print('\n')
            if args.dataset in ['cifar100', 'cifar10']:
                objs_avg = objs.avg
                top1_avg = top1.avg
                top5_avg = top5.avg
            elif args.dataset == 'ADP-Release1':
                objs_avg = objs.avg
                top1_avg = (top1.sum_accuracy.cpu().item() / (trained_data_size * n_classes))
                top5_avg = (top5.sum_accuracy.cpu().item() / trained_data_size) 
            
            logging.info('train %03d %e %f %f', step, objs_avg, top1_avg, top5_avg)

    if args.dataset in ['cifar100', 'cifar10']:
        objs_avg = objs.avg
        top1_avg = top1.avg
        top5_avg = top5.avg
    elif args.dataset == 'ADP-Release1':
        objs_avg = objs.avg
        top1_avg = (top1.sum_accuracy.cpu().item() / (len(train_queue.dataset) * n_classes))
        top5_avg = (top5.sum_accuracy.cpu().item() / len(train_queue.dataset)) 

    return top1_avg, top5_avg, objs_avg

def infer(valid_queue, model, criterion):
    global is_multi_gpu
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    # for ADP dataset
    preds = 0
    valided_data_size = 0
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            print('\rinfer mini batch {:03d}'.format(step), end=' ')

            input = input.cuda()
            target = target.cuda()

            logits = model(input)
            loss = criterion(logits, target)

            n = input.size(0)
            valided_data_size += n

            if args.dataset in ['cifar100', 'cifar10']:
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
            elif args.dataset == 'ADP-Release1':
                m = nn.Sigmoid()
                preds = (m(logits) > 0.5).int()
                prec1, prec5 = utils.accuracyADP(preds, target)
                objs.update(loss.item(), n)
                top1.update(prec1.double(), n)
                top5.update(prec5.double(), n)

            if step % args.report_freq == 0:
                print('\n')
                if args.dataset in ['cifar100', 'cifar10']:
                    objs_avg = objs.avg
                    top1_avg = top1.avg
                    top5_avg = top5.avg
                elif args.dataset == 'ADP-Release1':
                    objs_avg = objs.avg
                    top1_avg = (top1.sum_accuracy.cpu().item() / (valided_data_size * n_classes))
                    top5_avg = (top5.sum_accuracy.cpu().item() / valided_data_size) 
            
                logging.info('valid %03d %e %f %f', step, objs_avg, top1_avg, top5_avg)

    if args.dataset in ['cifar100', 'cifar10']:
        objs_avg = objs.avg
        top1_avg = top1.avg
        top5_avg = top5.avg
    elif args.dataset == 'ADP-Release1':
        objs_avg = objs.avg
        top1_avg = (top1.sum_accuracy.cpu().item() / (len(valid_queue.dataset) * n_classes))
        top5_avg = (top5.sum_accuracy.cpu().item() / len(valid_queue.dataset))

    return top1_avg, top5_avg, objs_avg


def write_data(epoch, net_metrics, lr_metrics, weights_normal, weights_reduce, genotype,
               perform_stat, arch_stat, genotype_stat, metrics_path, weights_path, genotypes_path):
    # genotype
    if epoch % 5 == 0 or epoch == args.epochs - 1:
        genotype_stat['epoch_{}'.format(epoch)] = [genotype]
        genotypes_df = pd.DataFrame(data=genotype_stat)
        genotypes_df.to_excel(genotypes_path)

    # io metrics
    perform_stat['S_epoch_{}'.format(epoch)] = net_metrics
    # perform_stat['out_S_epoch_{}'.format(epoch)] = net_metrics.output_channel_S
    # perform_stat['fc_S_epoch_{}'.format(epoch)] = net_metrics.fc_S
    # perform_stat['in_rank_epoch_{}'.format(epoch)] = net_metrics.input_channel_rank
    # perform_stat['out_rank_epoch_{}'.format(epoch)] = net_metrics.output_channel_rank
    # perform_stat['fc_rank_epoch_{}'.format(epoch)] = net_metrics.fc_rank
    # perform_stat['in_condition_epoch_{}'.format(epoch)] = net_metrics.input_channel_condition
    # perform_stat['out_condition_epoch_{}'.format(epoch)] = net_metrics.output_channel_condition
    if args.adas:
        # lr metrics
        # perform_stat['rank_velocity_epoch_{}'.format(epoch)] = lr_metrics.rank_velocity
        perform_stat['learning_rate_epoch_{}'.format(epoch)] = lr_metrics
    # write metrics data to xls file
    metrics_df = pd.DataFrame(data=perform_stat)
    metrics_df.to_excel(metrics_path)

    # weights
    # normal
    arch_stat['normal_none_epoch{}'.format(epoch)] = weights_normal[:, 0]
    arch_stat['normal_max_epoch{}'.format(epoch)] = weights_normal[:, 1]
    arch_stat['normal_avg_epoch{}'.format(epoch)] = weights_normal[:, 2]
    arch_stat['normal_skip_epoch{}'.format(epoch)] = weights_normal[:, 3]
    arch_stat['normal_sep_3_epoch{}'.format(epoch)] = weights_normal[:, 4]
    arch_stat['normal_sep_5_epoch{}'.format(epoch)] = weights_normal[:, 5]
    arch_stat['normal_dil_3_epoch{}'.format(epoch)] = weights_normal[:, 6]
    arch_stat['normal_dil_5_epoch{}'.format(epoch)] = weights_normal[:, 7]
    # reduce
    arch_stat['reduce_none_epoch{}'.format(epoch)] = weights_reduce[:, 0]
    arch_stat['reduce_max_epoch{}'.format(epoch)] = weights_reduce[:, 1]
    arch_stat['reduce_avg_epoch{}'.format(epoch)] = weights_reduce[:, 2]
    arch_stat['reduce_skip_epoch{}'.format(epoch)] = weights_reduce[:, 3]
    arch_stat['reduce_sep_3_epoch{}'.format(epoch)] = weights_reduce[:, 4]
    arch_stat['reduce_sep_5_epoch{}'.format(epoch)] = weights_reduce[:, 5]
    arch_stat['reduce_dil_3_epoch{}'.format(epoch)] = weights_reduce[:, 6]
    arch_stat['reduce_dil_5_epoch{}'.format(epoch)] = weights_reduce[:, 7]
    # write weights data to xls file
    weights_df = pd.DataFrame(data=arch_stat)
    weights_df.to_excel(weights_path)


if __name__ == '__main__':
    main()
