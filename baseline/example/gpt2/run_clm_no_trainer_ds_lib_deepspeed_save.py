#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import os
import math
from tqdm import tqdm

# from utils_model import get_network




import numpy as np
import matplotlib.pyplot as plt
import time
import timeit
import numpy as np

import uuid
import shutil
from typing import Dict, Optional
import torchsnapshot
from torchsnapshot import Snapshot, Stateful


from utils_gpt import get_argument_parser, \
    get_summary_writer, write_summary_events, \
    is_time_to_exit, check_early_exit_warning
    
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import deepspeed

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from multiprocessing import shared_memory


try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

import deepspeed_naive_lib as deepspeed_naive_lib
import deepspeed_naive_lib.utils

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.39.0.dev0")

# logger = get_logger(__name__)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)


logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    
    # Max Epochs
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    
    parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--fp16',
                        default=False,
                        # action='store_true',
                        help="Mixed precision training")
    # 
    parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
    
    
    parser.add_argument('--batches-per-commit', type=int, default=50,
                    help='number of batches processed before calling `state.commit()`; '
                         'commits prevent losing progress if an error occurs, but slow '
                         'down training.')
    parser.add_argument('--batches-per-host-check', type=int, default=5,
                    help='number of batches processed before calling `state.check_host_updates()`; '
                         'this check is very fast compared to state.commit() (which calls this '
                         'as part of the commit process), but because still incurs some cost due '
                         'to broadcast, so we may not want to perform it every batch.')

    
       
    parser.add_argument('--model-net', default='gpt_2', type=str, help='net type')
    
    parser.add_argument('--model', type=str, default='gpt_2',
                    help='model to benchmark')
    
    parser.add_argument('--num-warmup-batches', type=int, default=20,
                        help='number of warm-up batches that don\'t count towards benchmark')
    parser.add_argument('--num-batches-per-iter', type=int, default=10,
                        help='number of batches per benchmark iteration')
    parser.add_argument('--num-iters', type=int, default=50,
                        help='number of benchmark iterations')
    
    
    parser.add_argument('--mgwfbp', action='store_true', default=False, help='Use MG-WFBP')
    parser.add_argument('--asc', action='store_true', default=False, help='Use MG-WFBP')
    parser.add_argument('--nstreams', type=int, default=1, help='Number of communication streams')

    
    parser.add_argument('--threshold', type=int, default=34015396, help='Set threshold if mgwfbp is False')
    parser.add_argument('--rdma', action='store_true', default=False, help='Use RDMA')

    
    parser.add_argument('--compressor', type=str, default='topkef', help='Specify the compressors if density < 1.0')
    
    parser.add_argument('--memory', type=str, default = 'residual', help='Error-feedback')
    parser.add_argument('--density', type=float, default=0.01, help='Density for sparsification')
    
    parser.add_argument('--percent', type=float, default=0, help='percent of residual 0')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='gpt-large',help='model architecture')
    
    
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args





from enum import Enum
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

def evaluation():

    
    return


# 
# Calculate In-Memory Checkpoint time
# 
def calculate_in_memory_ckpt_time(model , optimizer,  idx):

    in_memory_time = time.time()
    _model_state_dict_cpu = {}
    numel_count = 0
    # Construct parameter state In-Memory Checkpoint scheme 

    for key, value in model.state_dict().items():
        t_cpu = torch.zeros(value.numel(), device='cpu', dtype=value.dtype, requires_grad=False)
        _model_state_dict_cpu[key] = t_cpu                    
        # Clone tensor
        value_clone = value.clone()
        # Save to CPU memory based on copy_
        _model_state_dict_cpu[key].copy_(value_clone.view(value.numel()), non_blocking=False)
        
        numel_count += value.numel()
        

    
    print('model_state_in_memory_time = ', time.time()- in_memory_time)


    in_memory_time = time.time()
    # Construct optimizer state In-Memory Checkpoint scheme 
    if optimizer.state_dict()['optimizer_state_dict']['state']!={} and True:
        exp_avg_0_numel = optimizer.state_dict()['optimizer_state_dict']['state'][0]['exp_avg'].numel()
        exp_avg_sq_0_numel = optimizer.state_dict()['optimizer_state_dict']['state'][0]['exp_avg_sq'].numel()

        exp_avg = optimizer.state_dict()['optimizer_state_dict']['state'][0]['exp_avg']
        exp_avg_cpu = torch.zeros(exp_avg_0_numel, device='cpu', dtype=exp_avg.dtype, requires_grad=False)
        exp_avg_cpu.copy_(exp_avg.view(exp_avg_0_numel), non_blocking=False)
        
        
        exp_avg_sq = optimizer.state_dict()['optimizer_state_dict']['state'][0]['exp_avg']
        exp_avg_sq_cpu = torch.zeros(exp_avg_sq_0_numel, device='cpu', dtype=exp_avg_sq.dtype, requires_grad=False)
        exp_avg_sq_cpu.copy_(exp_avg_sq.view(exp_avg_sq_0_numel), non_blocking=False)

        
                    
        
        
        

        fp32_flat_groups_0 = optimizer.state_dict()['fp32_flat_groups'][0]
        fp32_flat_groups_0_numel =fp32_flat_groups_0.numel()
        fp32_flat_groups_0_cpu = torch.zeros(fp32_flat_groups_0_numel, device='cpu', dtype=fp32_flat_groups_0.dtype, requires_grad=False)
        fp32_flat_groups_0_cpu.copy_(exp_avg_sq_cpu.view(fp32_flat_groups_0_numel), non_blocking=False) 
        
    print('optimizer_state_in_memory_time = ', time.time()- in_memory_time)
    
    return

    

            
            # 
             
            # 
            
    pass




def full_train():
    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()
    
    load_state_dict_time = time.time()
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        # accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    # total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    total_batch_size = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
   
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)
        
        
    
    if torch.distributed.get_rank() == 0:
        print('load_state_dict_time = ', time.time() - load_state_dict_time)
    
    # update the progress_bar if load from checkpoint
    # progress_bar.update(completed_steps)
    verbose = 1 if torch.distributed.get_rank() == 0 else 0
    
    if torch.distributed.get_rank() == 0:
        print("Start training!!!")
        print("starting_epoch: ", starting_epoch)
        print("args.num_train_epochs: ", args.num_train_epochs)
        print("args.max_train_steps: ", args.max_train_steps)

    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # print('device = ', device)
    # model.to(device)
    print_steps = max(10, args.ckpt_freq)
    train_start = time.time()
    iteration_time = AverageMeter('Time', ':6.3f')
    batch_time_start = time.time()
    for epoch in range(starting_epoch, args.num_train_epochs):
        io_time_array = []
        forward_backforward_time_array = []
        forward_time_array  = []
        backward_time_array = [] 
        
        communication_time_array  = []
        step_time_array    = []
        elastic_time_array = []
        batch_time_array = []
        
        optimizer.synchronize_time= []
        
        train_sampler.set_epoch(epoch) 

        model.train()
        with tqdm(total=len(train_dataloader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
            if args.with_tracking:
                total_loss = 0

            # if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            #     # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            #     active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            # else:
            #     active_dataloader = train_dataloader

            active_dataloader = train_dataloader
            end = time.time()
            for step, batch in enumerate(active_dataloader):

                e_time = time.time()
                # state.commit()
                # state.check_host_updates()
                # elastic_time_array.append(time.time()-e_time)
                
                
                s_time = time.time()   
                # with accelerator.accumulate(model):
                ### to cuda
                batch = {key: value.to(device) for key, value in batch.items()}
                
                f_time = time.time()
                # batch = batch.to(device)
                outputs = model(**batch)
                
                loss = outputs.loss
                
                # We keep track of the loss at each epoch
                # if args.with_tracking:
                #     total_loss += loss.detach().float()
                
                # accelerator.backward(loss)
                forward_time_array.append(time.time() - f_time)
                
                b_time = time.time()
                
                # loss.backward()
                model.backward(loss)
                
                
                backward_time_array.append(time.time() - b_time)
                forward_backforward_time_array.append(time.time()-s_time)
                
                st_time = time.time()
                # optimizer.step()
                model.step()
                
                step_time_array.append(time.time()-st_time)
                
                lr_scheduler.step()
                optimizer.zero_grad()
                
                
                    
                
                t.update(1)
                completed_steps += 1
                if  completed_steps % args.ckpt_freq == 0:
                    state = {
                            'epoch': epoch + 1,
                            'iteration': completed_steps + 1,
                            'arch': args.arch
                            # 'state_dict': model.state_dict(),
                            # 'best_acc1': best_acc1,
                            # 'optimizer' : optimizer.state_dict(),
                            # 'scheduler' : scheduler.state_dict()
                        }
                    deepspeed_naive_lib.utils.save_checkpoint_iteration_deepspeed(model, state, epoch + 1,  completed_steps)
                batch_time_array.append(time.time()-e_time)
                if step >= 30: 
                    iteration_time.update(time.time() - end)
                

                # Checks if the accelerator has performed an optimization step behind the scenes
                # if accelerator.sync_gradients:
                #     progress_bar.update(1)
                #     completed_steps += 1
                    
                
                if completed_steps % print_steps == 0 and torch.distributed.get_rank()==0:
                    
                    print(f'-----------------Step=%d----------------'%step)
                    
                    # forward_backforward_time =sum(forward_backforward_time_array)
                    forward_time     = sum(forward_time_array)
                    backward_time    = sum(backward_time_array)
                    elastic_time     = sum(elastic_time_array)
                    batch_time       = sum(batch_time_array)
                    
                    print('elastic_time  = ', elastic_time)
                    print('forward_time  = ', forward_time)
                    print('backward_time = ', backward_time)
                    
                    # print('forward_backforward_time = ', forward_backforward_time)
                    print('batch_time = ', batch_time)
                    print('Avg_Iteartion_Time = ', iteration_time.avg)
                    
                    batch_time_end = time.time() - batch_time_start

                    
                    
                    print('per_batch_time = ', batch_time_end/print_steps)
                    
                    print('per_backward_time = ', backward_time/print_steps)
                    
                    print('------------------------------------------------')
                    # calculate_in_memory_ckpt_time(model , optimizer,  step)
                    
                    
                    batch_time_start = time.time()
                     
                    io_time_array = []
                    forward_backforward_time_array = []
                    forward_time_array  = []
                    backward_time_array = []

                    communication_time_array  = []
                    step_time_array    = []
                    elastic_time_array = []   
                    batch_time_array   =[]  
                    
                    while len(ckpt_path) > 1:
                        del_filepath = ckpt_path[0]
                        if os.path.exists(del_filepath):
                            os.remove(del_filepath)
                        del ckpt_path[0]
             
                
                checkpointing_steps = 100
                
                
                
                if isinstance(checkpointing_steps, int) and torch.distributed.get_rank()==0:
                    
                    save_checkpoint_time = time.time()
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"epoch_{epoch}_step_{completed_steps}.bin"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)

                        

                        print('save_checkpoint_time = ', time.time() - save_checkpoint_time)


                


                if completed_steps >= args.max_train_steps:
                    break
                
                end = time.time()
                    
        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            ### to cuda
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            # losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
            losses.append(loss)

        # Convert 0-dimensional scalars in losses to 1-dimensional tensors
        losses = [loss.unsqueeze(0) for loss in losses]
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
        
        

        
        
    if torch.distributed.get_rank() == 0:
        print("Final_Iteration_Time = ", iteration_time.avg)
        with open("./iteartion_time_bert.txt", "w+") as fp:
            fp.write(iteration_time.avg)
    return



# Delete folder contents
def delete_folder_contents(folder):
    # filenames = os.listdir(folder)
    # if len(filenames) < 1000:
    #     return
    
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    return

        

ckpt_path = []

# def main():
if __name__ == "__main__":

    print('-----------------args------------------')

    # args = parse_args()
    parser = get_argument_parser()

    deepspeed.init_distributed(dist_backend='nccl')

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    args.checkpoint_format = "./gpt2_checkpoint/gpt2_epoch{epoch}_iteration{iteration}"
    args.ckpt_mode = 1
    
    check_early_exit_warning(args)

    # if args.local_rank == -1 or args.no_cuda:
    #     device = torch.device("cuda" if torch.cuda.is_available()
    #                           and not args.no_cuda else "cpu")
    #     n_gpu = torch.cuda.device_count()
    # else:
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     n_gpu = 1
    #     # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.distributed.init_process_group(backend='nccl')
    # logger.info(
    #     "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".
    #     format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size /
                                args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")



    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # # in the environment
    
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir


    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    # logger.info(accelerator.state, main_process_only=False)
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(args.seed)

    # Handle the repository creation
    # if accelerator.is_main_process:
    #     if args.push_to_hub:
    #         # Retrieve of infer repo_name
    #         repo_name = args.hub_model_id
    #         if repo_name is None:
    #             repo_name = Path(args.output_dir).absolute().name
    #         # Create repo and retrieve repo_id
    #         api = HfApi()
    #         repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

    #         with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
    #             if "step_*" not in gitignore:
    #                 gitignore.write("step_*\n")
    #             if "epoch_*" not in gitignore:
    #                 gitignore.write("epoch_*\n")
    #     elif args.output_dir is not None:
    #         os.makedirs(args.output_dir, exist_ok=True)
    # accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    download_load_dataset_time = time.time()
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    if torch.distributed.get_rank() == 0:
        print('download_load_dataset_time = ', time.time()-download_load_dataset_time)
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.
    
    init_model_time = time.time()
    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        # print('args.config_name')
        
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        # 
        if torch.distributed.get_rank() == 0:
            print('args.model_name_or_path')
        
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        if torch.distributed.get_rank() == 0:
            print('max_position_embeddings = ', config.max_position_embeddings)
        
    else:
        # print('else model_name_or_path')

        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")


    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    
    if torch.distributed.get_rank() == 0:
        print('init_model_time = ', time.time()-init_model_time)
    
    
    # Preprocessing the datasets.
    process_data_time = time.time()
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    
    args.preprocessing_num_workers = 16
    
    

    
    tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)

            if torch.distributed.get_rank() == 0:
                print('block_size = min(1024, config.max_position_embeddings)')

    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and 
    # generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    # with accelerator.main_process_first():
    #     lm_datasets = tokenized_datasets.map(
    #         group_texts,
    #         batched=True,
    #         num_proc=args.preprocessing_num_workers,
    #         load_from_cache_file=not args.overwrite_cache,
    #         desc=f"Grouping texts in chunks of {block_size}",
    # )

    
    lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank())    
    train_dataloader = DataLoader(
        train_dataset,  collate_fn = default_data_collator, batch_size=args.per_device_train_batch_size,
        sampler=train_sampler, **kwargs)

    # DataLoaders creation:
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    # )
    # eval_dataloader = DataLoader(
    #     eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    # )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset, num_replicas=torch.distributed.get_world_size(), rank=torch.distributed.get_rank())
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size,
        sampler=val_sampler, **kwargs)

    if torch.distributed.get_rank() == 0:
        print('process_data_time = ', time.time() - process_data_time)

    # Optimizer
    init_hvd_time = time.time()
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if torch.distributed.get_rank() == 0:
        print("---args.num_train_epochs: ", args.num_train_epochs)
        print("---args.max_train_steps: ", args.max_train_steps)
        print("---len(train_dataloader): ", len(train_dataloader))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        # num_warmup_steps = args.num_warmup_steps * accelerator.num_processes,
        num_warmup_steps = args.num_warmup_steps * torch.distributed.get_world_size(),
        num_training_steps=args.max_train_steps
        
        # if overrode_max_train_steps
        # else args.max_train_steps * accelerator.num_processes,
    )

    ### model to cuda
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(device)

    # model, optimizer, _, _ = deepspeed.initialize(
    #     args=args,
    #     model=model,
    #     model_parameters=optimizer_grouped_parameters,
    #     dist_init_required=True)
    
    
    
    model, optimizer, _, _ = deepspeed_naive_lib.initialize(
        args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters,
        dist_init_required=True)


    if torch.distributed.get_rank() == 0:
        print('init_hvd_time =', time.time() - init_hvd_time)
     
    state_time = time.time()
    
    resume_from_epoch = 0
    
    if torch.distributed.get_rank() == 0:
        print('state_time = ', time.time() - state_time)
    
    
    
    

    
    
    
    
    if torch.distributed.get_rank()==0:
        print('state_time = ', time.time() - state_time)
    
    _state_dict_cpu = {}
    numel_count = 0
    for key, value in model.state_dict().items():
        t_cpu = torch.zeros(value.numel(), device='cpu', dtype=value.dtype, requires_grad=False)
        _state_dict_cpu[key] = t_cpu
        numel_count += value.numel()
    
    _state_dict_gpu_flatten = torch.zeros(numel_count, device=value.device, dtype=value.dtype, requires_grad=False)
    
    if torch.distributed.get_rank() == 0:
        
        
        print('optimizer.state_dict() = ', optimizer.state_dict().keys())
        
        pass

    # torchsnapshot
    progress = torchsnapshot.StateDict(current_epoch=0)
    # torchsnapshot: define app state
    app_state: Dict[str, Stateful] = {
        "rng_state": torchsnapshot.RNGState(),
        "model": model,
        "optim": optimizer,
        "progress": progress,
    }
    snapshot: Optional[Snapshot] = None
    checkpoint_save_work_dir = './gpt2_checkpoint'



    full_train()


