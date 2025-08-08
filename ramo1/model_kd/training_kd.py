import torch
import time
import os
import logging
from tqdm import tqdm
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from utils import AverageMeter, calculate_accuracy, save_checkpoint, save_dict_to_json, load_checkpoint
from evaluate import evaluate_kd


def train_epoch_kd(epoch,
                   data_loader,
                   model,
                   teacher_model,
                   optimizer,
                   loss_fn_kd,
                   params,
                   epoch_logger=None,
                   batch_logger=None,
                   tb_writer=None):
    logging.info(f"[Distillazione] Inizio epoca {epoch}")
    model.train()
    teacher_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    with tqdm(total=len(data_loader)) as t:
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.permute(0, 2, 1, 3, 4).to(params.device)  # [B, C, T, H, W]
            targets = targets.to(params.device, non_blocking=True)
            data_time.update(time.time() - end_time)

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            outputs = model(inputs)
            loss = loss_fn_kd(outputs, targets, teacher_outputs, params)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_logger is not None:
                batch_logger.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * len(data_loader) + (i + 1),
                    'loss': losses.val,
                    'acc': accuracies.val,
                    'lr': params.learning_rate
                })

            t.set_postfix(loss=losses.avg, acc=accuracies.avg)
            t.update(1)

            logging.info(f"Epoch [{epoch}][{i+1}/{len(data_loader)}] "
                         f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | "
                         f"Data {data_time.val:.3f} ({data_time.avg:.3f}) | "
                         f"Loss {losses.val:.4f} ({losses.avg:.4f}) | "
                         f"Acc {accuracies.val:.3f} ({accuracies.avg:.3f})")

    if getattr(params, "distributed", False):
        loss_sum = torch.tensor([losses.sum], device=params.device)
        loss_count = torch.tensor([losses.count], device=params.device)
        acc_sum = torch.tensor([accuracies.sum], device=params.device)
        acc_count = torch.tensor([accuracies.count], device=params.device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': params.learning_rate
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('train/lr', params.learning_rate, epoch)

    logging.info(f"[Distillazione] Fine epoca {epoch} — Loss: {losses.avg:.4f}, Acc: {accuracies.avg:.2f}%")


def train_and_evaluate_kd(model,
                          teacher_model,
                          train_dataloader,
                          val_dataloader,
                          optimizer,
                          loss_fn_kd,
                          metrics,
                          params,
                          model_dir,
                          restore_file=None):
    """
    Addestra il modello con distillazione e valuta ad ogni epoca.
    """
    os.makedirs(model_dir, exist_ok=True)

    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info(f"Ripristino checkpoint da {restore_path}")
        load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "cnn_distill":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    else:
        scheduler = None

    for epoch in range(params.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{params.num_epochs}")

        train_epoch_kd(
            epoch=epoch,
            data_loader=train_dataloader,
            model=model,
            teacher_model=teacher_model,
            optimizer=optimizer,
            loss_fn_kd=loss_fn_kd,
            params=params,
            epoch_logger=None,
            batch_logger=None,
            tb_writer=None
        )

        val_metrics = evaluate_kd(model, val_dataloader, metrics, params)

        if scheduler is not None:
            scheduler.step()

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict()
        }, is_best=is_best, checkpoint=model_dir)

        if is_best:
            logging.info(f"✅ Nuova miglior accuratezza: {val_acc:.4f}")
            best_val_acc = val_acc
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)
