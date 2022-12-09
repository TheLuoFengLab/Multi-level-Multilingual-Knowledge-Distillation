# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel

import mmkd.builder
import mmkd.even_paralleldataset
import transformers

transformers.logging.set_verbosity_error()

UNK, SEP, PAD, CLS, MASK = "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"
m_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
special_token_id_list = m_tokenizer.convert_tokens_to_ids([UNK, SEP, PAD, CLS, MASK])
unk_id, sep_id, pad_id, cls_id, mask_id = special_token_id_list
VOCAB_SIZE = m_tokenizer.vocab_size

parser = argparse.ArgumentParser(description='MMKD mBERT Pre-Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
# output dim
parser.add_argument('--mmkd-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
# hidden dim
parser.add_argument('--mmkd-mlp-dim', default=768, type=int,
                    help='hidden dimension in MLPs (default: 768)')
parser.add_argument('--mmkd-t', default=0.05, type=float,
                    help='softmax temperature (default: 0.05)')

# other upgrades
parser.add_argument('--optimizer', default='adamw', type=str,
                    # choices=['lars', 'adamw'],
                    help='optimizer used (default: adamw)')
parser.add_argument('--warmup-epochs', default=2, type=int, metavar='N',
                    help='number of warmup epochs')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

# gpu is process id
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        # tcp initialization
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        # Synchronizes all processes.
        # This collective blocks processes until the whole group enters this function, if async_op is False, or if async work handle is called on wait()
        torch.distributed.barrier()
    # create model
    print("=> creating model--------------")
    model = mmkd.builder.MMKD_mBERT(args.mmkd_dim, args.mmkd_mlp_dim, args.mmkd_t)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('--------params num---------',params)
    
    base_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    momentum_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # infer learning rate before changing batch size
    # use larger lr to accelerate converge when using larger batch size
    args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # the process will in this statement in our case
        # apply SyncBN
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            # process will be here
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm
    optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    # To prevent underflow, “gradient scaling” multiplies the network’s loss(es) by a scale factor and invokes a backward pass on the scaled loss(es). 
    # Gradients flowing backward through the network are then scaled by the same factor. 
    # In other words, gradient values have a larger magnitude, so they don’t flush to zero.    
    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if args.rank == 0 else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    train_dataset = mmkd.even_paralleldataset.ParallelcorpusDataset('/vast_data/mingqil/clean_data/') 

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, base_tokenizer, momentum_tokenizer, optimizer, scaler, summary_writer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, is_best=False, filename='/vast_data/mingqil/mmkd_ckp/checkpoint_%04d.pth.tar' % epoch)

    if args.rank == 0:
        summary_writer.close()

def train(train_loader, model, base_tokenizer, momentum_tokenizer, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    # moco_m = args.moco_m
    for i, sen in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        encoding_tlm = batchTLM(sen,base_tokenizer)
        input_ids_tlm = encoding_tlm['input_ids']
        token_type_ids_tlm = encoding_tlm['token_type_ids']
        attention_mask_tlm = encoding_tlm['attention_mask']
        labels_tlm = encoding_tlm['labels']
        truth_en, inp_en, seg_en, attn_msk_en, contrastive_labels, wlist_en, wlist_mul, seg_mul, attn_msk_mul = batchTaCL(sen,base_tokenizer,momentum_tokenizer,args)

        encoding_dict1 = base_tokenizer.batch_encode_plus(sen[1],max_length=128,pad_to_max_length=True,return_tensors='pt')
        encoding_dict2 = momentum_tokenizer.batch_encode_plus(sen[0],max_length=128,pad_to_max_length=True,return_tensors='pt')
        
        input_ids1 = encoding_dict1['input_ids']
        input_ids2 = encoding_dict2['input_ids']

        token_type_ids1 = encoding_dict1['token_type_ids']
        token_type_ids2 = encoding_dict2['token_type_ids']

        attention_mask1 = encoding_dict1['attention_mask']
        attention_mask2 = encoding_dict2['attention_mask']

        if args.gpu is not None:
            input_ids1 = input_ids1.cuda(args.gpu, non_blocking=True)
            input_ids2 = input_ids2.cuda(args.gpu, non_blocking=True)
            token_type_ids1 = token_type_ids1.cuda(args.gpu, non_blocking=True)
            token_type_ids2 = token_type_ids2.cuda(args.gpu, non_blocking=True)
            attention_mask1 = attention_mask1.cuda(args.gpu, non_blocking=True)
            attention_mask2 = attention_mask2.cuda(args.gpu, non_blocking=True)
            input_ids_tlm, token_type_ids_tlm, attention_mask_tlm, labels_tlm = input_ids_tlm.cuda(args.gpu), token_type_ids_tlm.cuda(args.gpu), attention_mask_tlm.cuda(args.gpu), labels_tlm.cuda(args.gpu)
            truth_en, inp_en, seg_en, attn_msk_en, contrastive_labels, seg_mul, attn_msk_mul = truth_en.cuda(args.gpu), inp_en.cuda(args.gpu), seg_en.cuda(args.gpu), attn_msk_en.cuda(args.gpu), contrastive_labels.cuda(args.gpu), seg_mul.cuda(args.gpu), attn_msk_mul.cuda(args.gpu)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(input_ids1, input_ids2, token_type_ids1, token_type_ids2, attention_mask1, attention_mask2, truth_en, inp_en, seg_en, attn_msk_en, contrastive_labels, wlist_en, wlist_mul, input_ids_tlm, token_type_ids_tlm, attention_mask_tlm, labels_tlm, seg_mul, attn_msk_mul) 
        losses.update(loss.item(), input_ids1.size(0))
        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        # 0.5lr -> lr -> 0.5lr
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def random_token(tokenizer):
    rand_idx = 1 + np.random.randint(VOCAB_SIZE-1)
    while rand_idx in special_token_id_list:
        rand_idx = 1 + np.random.randint(VOCAB_SIZE-1)
    random_token = tokenizer.convert_ids_to_tokens([rand_idx])[0]
    return random_token


def random_mask(input_ids):
    tokenlen = torch.count_nonzero(input_ids,dim=1)
    bsz, seqlen = input_ids.size()
    for i in range(bsz):
        rand = torch.rand(tokenlen[i])
        mask_arr = (torch.cat((rand<0.15,torch.zeros(seqlen-tokenlen[i],dtype=torch.bool)),dim=0)) * (input_ids[i]!=101) * (input_ids[i]!=102)
        selection = torch.flatten((mask_arr.nonzero())).tolist()
        for j in selection:
            if random.random()<0.8:
                input_ids[i,j] = 103
            else:
                if random.random()<0.5:
                    pass
                else:
                    random_idx = 1 + np.random.randint(VOCAB_SIZE-1)
                    while random_idx in special_token_id_list:
                        random_idx = 1 + np.random.randint(VOCAB_SIZE-1)
                    input_ids[i,j] = random_idx
    return input_ids


def whole_word_random_mask(tokens, masked_lm_prob, tokenizer):
    joined_word_list = []
    for token in tokens:
        if token.startswith('##'):
            assert len(joined_word_list) != 0
            joined_word_list[-1].append(token)
        else:
            joined_word_list.append([token])
    cand_idx_list = []
    for idx in range(len(joined_word_list)):
        item = joined_word_list[idx]
        if len(item)==1 and item[0] in [UNK, SEP, PAD, CLS, MASK]:
            pass
        else:
            cand_idx_list.append(idx)
    num_words = len(cand_idx_list)
    num_to_predict = min(int(num_words*0.25), max(1, int(round(num_words*masked_lm_prob))))
    random.shuffle(cand_idx_list)
    cand = set(cand_idx_list[:num_to_predict])
    masked_tokens, mask, tgt = [], [], []
    for i, one_word_list in enumerate(joined_word_list):
        if i in cand:
            for token in one_word_list:
                if random.random() < 0.8:
                    masked_tokens.append(MASK)
                else:
                    if random.random() < 0.5:
                        masked_tokens.append(token)
                    else:
                        masked_tokens.append(random_token(tokenizer))
                mask.append(1)
                tgt.append(token)
        else:
            for token in one_word_list:
                masked_tokens.append(token)
                mask.append(0)
                tgt.append(PAD)
    return masked_tokens, mask, tgt


def ListsToTensor(xs,tokenize=False,tokenizer=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if tokenize:
            y = tokenizer.convert_tokens_to_ids(x) + [pad_id]*(max_len-len(x))
        else:
            y = x + [0]*(max_len-len(x))
        ys.append(y)
    data = torch.LongTensor(ys).contiguous()
    return data


def batchTLM(data,tokenizer):
    pairs = []
    for i in range(len(data[0])):
        x = [data[0][i],data[1][i]]
        pairs.append(x)
    inputs = tokenizer.batch_encode_plus(pairs,max_length=128,pad_to_max_length=True,return_tensors='pt')
    inputs['labels'] = inputs.input_ids.detach().clone()
    inputs.input_ids = random_mask(inputs.input_ids)
    return inputs


def token_to_word(tokens):
    c = -1
    wlist = []
    for i in range(len(tokens)):
        if tokens[i].startswith('##'):
            wlist.append(c)
        else:
            c += 1
            wlist.append(c)
    return wlist


def process_tgt(tgt_matrix,tokenizer):
    # only masked token is token, others are PAD in tgt
    max_len = max(len(x) for x in tgt_matrix)
    ys = []
    # y is [0,0,0,0,masked_token_id,0,0]
    for x in tgt_matrix:
        y = tokenizer.convert_tokens_to_ids(x) + [pad_id]*(max_len-len(x))
        ys.append(y)
    ys = torch.LongTensor(ys).contiguous()
    contrastive_labels = ys.clone()
    contrastive_labels[contrastive_labels[:,:]==pad_id] = 0
    contrastive_labels[contrastive_labels[:,:]!=pad_id] = 1
    return contrastive_labels


def batchTaCL(data,tokenizer,momentum_tokenizer,args):
    truth, inp, seg, tgt_matrix, wlist_en, wlist_mul, seg_mul, attn_msk_mul = [], [], [], [], [], [], [], []
    for i in range(args.batch_size):
        # input into BERT using Bert tokenizer
        a = momentum_tokenizer.tokenize(data[0][i])
        x = [CLS]+a+[SEP]
        w_en = token_to_word(x)
        wlist_en.append(w_en)
        # truth is CLS+english+SEP
        truth.append(x)
        seg.append([0]*(len(a)+2))
        # input into mBERT using mBert tokenizer
        b = tokenizer.tokenize(data[0][i])
        y = [CLS]+b+[SEP]
        w_mul = token_to_word(y)
        wlist_mul.append(w_mul)
        masked_y, mask, tgt = whole_word_random_mask(y, 0.15, tokenizer)
        c = tokenizer.tokenize(data[1][i])
        masked_y = masked_y+c+[SEP]
        # input inp into mBERT, input truth into BERT
        inp.append(masked_y)
        seg_mul.append([0]*(len(b)+2)+[1]*(len(c)+1))
        tgt_matrix.append(tgt)

    truth = ListsToTensor(truth, tokenize=True, tokenizer=momentum_tokenizer)
    inp = ListsToTensor(inp, tokenize=True, tokenizer=tokenizer)
    seg = ListsToTensor(seg, tokenize=False)
    attn_msk = ~truth.eq(pad_id)
    contrastive_labels = process_tgt(tgt_matrix,tokenizer)
    seg_mul = ListsToTensor(seg_mul, tokenize=False)
    attn_msk_mul = ~inp.eq(pad_id)
    return truth, inp, seg, attn_msk, contrastive_labels, wlist_en, wlist_mul, seg_mul, attn_msk_mul


if __name__ == '__main__':
    main()
