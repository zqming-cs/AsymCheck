import torch 
import os
import time
import uuid
import torch.distributed as dist
import torchsnapshot

def save_checkpoint_iteration(state, epoch,  iteration, save_dir='./checkpoint'):
    
    filename ='ckpt_epoch_' + str(epoch) + '_iteration_' + str(iteration)+ '_rank_' + str(dist.get_rank()) + '.pth.tar'
    save_file_path = os.path.join(save_dir, filename)
    torch.save(state, save_file_path)


def save_checkpoint_iteration_deepspeed(model, state, epoch,  iteration, save_dir = './checkpoint'):
    
    filename ='ckpt_epoch_' + str(epoch) + '_iteration_' + str(iteration)+ '_rank_' + str(dist.get_rank()) + '.pth.tar'
    save_file_path = os.path.join(save_dir, filename)
    model.save_checkpoint(save_dir= save_file_path, client_state=state)


def save_checkpoint_in_disk_snapshot(progress_save, app_state, checkpoint_save_work_dir='./checkpoint'):
    progress_save["current_epoch"] += 1
    snapshot = torchsnapshot.Snapshot.take(
        f"{checkpoint_save_work_dir}/run-{uuid.uuid4()}-epoch-{progress_save['current_epoch']}-model-optimizer",
        app_state,
        replicated=["**"],
        # this pattern treats all states as replicated
    )
    return

def calculate_in_memory_ckpt_time(model , optimizer,  idx):
    in_memory_time = time.time()
    _model_state_dict_cpu = {}
    numel_count = 0
    for key, value in model.state_dict().items():
        t_cpu = torch.zeros(value.numel(), device='cpu', dtype=value.dtype, requires_grad=False)
        _model_state_dict_cpu[key] = t_cpu                    
        value_clone = value.clone()

        _model_state_dict_cpu[key].copy_(value_clone.view(value.numel()), non_blocking=False)
        numel_count += value.numel()

    print('model_state_in_memory_time = ', time.time()- in_memory_time)


    in_memory_time = time.time()
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


def ei_sort_key(s):
    parts = s.split('-')
    try:
        epoch_index = parts.index('epoch')
        iteration_index = parts.index('iteration')
    
        epoch_value = int(parts[epoch_index + 1])
        iteration_value = int(parts[iteration_index + 1])
        return epoch_value, iteration_value
    except (ValueError, IndexError) as e:
        return int(-1), int(-1)

def load_shard_checkpoint(model, optimizer, numel, filedir='./checkpoint'):
    files  = os.listdir(filedir)
    files_sorted = sorted(files, key=ei_sort_key) 
    ckpt_to_load = os.path.join(filedir, files_sorted[-1])
    _, i = ei_sort_key(ckpt_to_load)
    tensor= torch.load(ckpt_to_load)['Optimizer']
    flag = False
    for param_group in optimizer.param_groups:
        keys = list(param_group.keys())
        if "momentum" in keys:
            new_tensor = tensor[numel["momentum"][0] : numel["momentum"][1]]
            param_group['params'] = torch.from_numpy(new_tensor).cuda()
        elif not flag:
            new_tensor = tensor[numel["exp_avg"][0] : numel["exp_avg"][1]]
            param_group['params'] = torch.from_numpy(new_tensor).cuda()
            flag = True
            continue
        elif flag:
            new_tensor = tensor[numel["exp_avg_sq"][0] : numel["exp_avg_sq"][1]]
            param_group['params'] = torch.from_numpy(new_tensor).cuda()
    print("loaded interation {}".format(i))
    return model, optimizer

def save_checkpoint_async_model(model, optimizer_state, epoch, idx, checkpoint_save_work_dir='./checkpoint'):
    async_time_array=[]
    async_time = time.time()
    output_model_file = os.path.join(checkpoint_save_work_dir, f"run-{uuid.uuid4()}-epoch-{epoch}-iteration-{idx}")
    model._process_model_in_memory(optimizer_state, 0, output_model_file)
    end_time = time.time() - async_time
    async_time_array.append(end_time)
    
    if dist.get_rank() == 0 :
        print('save_checkpoint_async = ', end_time)