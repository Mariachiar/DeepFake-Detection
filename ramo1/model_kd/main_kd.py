from pathlib import Path
import json
import random
import os
import numpy as np
import torch
from torch.backends import cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from modelTeacher.classifier.i3d_ori import Classifier
import torch
from opts_kd import parse_opts
from model import generate_model, load_pretrained_model, make_data_parallel, get_fine_tuning_parameters, loss_fn_kd
from train_dataset import FaceForensicsClipDataset
from training_kd import train_and_evaluate_kd
from utils import get_lr, worker_init_fn, save_checkpoint
# Inizializza la configurazione (i3d_ori.yaml)
from config import config as my_cfg

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()

    if opt.result_path is not None:
        os.makedirs(opt.result_path, exist_ok=True)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)


    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = f"{opt.model}-{opt.model_depth}"
    return opt


def get_train_loader(opt):
    dataset = FaceForensicsClipDataset(root_dir=opt.train_data_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if opt.distributed else None
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=(sampler is None),
                                         num_workers=opt.n_threads,
                                         pin_memory=True,
                                         worker_init_fn=worker_init_fn)
    return loader, sampler


def get_val_loader(opt):
    dataset = FaceForensicsClipDataset(root_dir=opt.val_data_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if opt.distributed else None
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size // opt.n_val_samples,
                                         shuffle=(sampler is None),
                                         num_workers=opt.n_threads,
                                         pin_memory=True,
                                         worker_init_fn=worker_init_fn)
    return loader


def main_worker(rank, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + rank
        dist.init_process_group(backend='nccl', init_method=opt.dist_url, world_size=opt.world_size, rank=opt.dist_rank)
        opt.batch_size //= opt.ngpus_per_node
        opt.n_threads = (opt.n_threads + opt.ngpus_per_node - 1) // opt.ngpus_per_node

    opt.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    model = generate_model(opt)

    if opt.pretrain_path:
        model = load_pretrained_model(model, opt.pretrain_path, opt.model, opt.n_finetune_classes)

    model = make_data_parallel(model, opt.distributed, opt.device)
    parameters = get_fine_tuning_parameters(model, opt.ft_begin_module) if opt.pretrain_path else model.parameters()

    my_cfg.init_with_yaml()
    my_cfg.update_with_yaml("i3d_ori.yaml")  # Assicurati che il file sia accessibile
    my_cfg.freeze()

    # Crea il modello teacher
    teacher_model = Classifier()
    teacher_model = teacher_model.to(opt.device)
    teacher_model.eval()

    # Carica i pesi
    state_dict = torch.load(opt.teacher_pretrain_path, map_location=opt.device)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    teacher_model.load_state_dict(state_dict, strict=False)

    if opt.teacher_path:
        checkpoint = torch.load(opt.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['state_dict'])

    train_loader, train_sampler = get_train_loader(opt)
    val_loader = get_val_loader(opt)

    optimizer = torch.optim.SGD(parameters, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.multistep_milestones)

    train_and_evaluate_kd(model, teacher_model, train_loader, val_loader, optimizer, loss_fn_kd,
                          metrics=opt.metrics, params=opt, model_dir=opt.result_path, restore_file=opt.resume_path)


if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True

    opt.ngpus_per_node = torch.cuda.device_count()
    if opt.distributed:
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    else:
        main_worker(0, opt)
