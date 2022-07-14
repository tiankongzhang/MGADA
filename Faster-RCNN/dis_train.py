import argparse
import datetime
import math
import os
import time

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from detection import utils
from detection.config import cfg
from detection.data.build import build_data_loaders
from detection.engine.eval import evaluation
from detection.modeling.build import build_detectors
from detection.utils import dist_utils

global_step = 0
total_steps = 0
best_mAP = -1.0


def cosine_scheduler(eta_max, eta_min, current_step):
    y = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(current_step / total_steps * math.pi))
    return y


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def box_to_centers(boxes):
    x = boxes[:, 2] - boxes[:, 0]
    y = boxes[:, 3] - boxes[:, 1]
    centers = torch.stack((x, y), dim=1)
    return centers


def detach_features(features):
    if isinstance(features, torch.Tensor):
        return features.detach()
    return tuple([f.detach() for f in features])


def convert_sync_batchnorm(model):
    convert = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            convert = True
            break
    if convert:
        print('Convert to SyncBatchNorm')
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def train_one_epoch(model, optimizer, train_loader, target_loader, device, epoch, dis_model, dis_optimizer, print_freq=10, writer=None, test_func=None, save_func=None):
    global global_step
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.0e}'))
    metric_logger.add_meter('LAMBDA', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_schedulers = []
    if epoch == 0:
        warmup_factor = 1. / 500
        warmup_iters = min(500, len(train_loader) - 1)
        # lr_schedulers = [
        #     warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor),
        #     warmup_lr_scheduler(dis_optimizer, warmup_iters, warmup_factor),
        # ]

    target_loader_iter = iter(target_loader)
    for images, img_metas, targets in metric_logger.log_every(train_loader, print_freq, header):
        global_step += 1
        images = images.to(device)
        targets = [t.to(device) for t in targets]

        try:
            t_images, t_img_metas, _ = next(target_loader_iter)
        except StopIteration:
            target_loader_iter = iter(target_loader)
            t_images, t_img_metas, _ = next(target_loader_iter)

        t_images = t_images.to(device)

        loss_dict, outputs = model(images, img_metas, targets, t_images, t_img_metas)
        adv_loss = loss_dict.pop('adv_loss')
        #print(adv_loss)
        loss_dict_for_log = dict(**loss_dict, **adv_loss)

        det_loss = sum(list(loss_dict.values()))
        ada_loss = sum(list(adv_loss.values()))

        LAMBDA = cosine_scheduler(cfg.ADV.LAMBDA_FROM, cfg.ADV.LAMBDA_TO, global_step)
        #print(LAMBDA)
        losses = det_loss + ada_loss * LAMBDA
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict_for_log)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(LAMBDA=LAMBDA)

        if global_step % print_freq == 0:
            if writer:
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar('losses/{}'.format(k), v, global_step=global_step)
                writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
                writer.add_scalar('LAMBDA', LAMBDA, global_step=global_step)

        if global_step % (2000 // max(1, (dist_utils.get_world_size() // 2))) == 0 and test_func is not None:
            updated = test_func()
            if updated:
                save_func('best.pth', 'mAP: {:.4f}'.format(best_mAP))
            print('Best mAP: {:.4f}'.format(best_mAP))


def main(cfg, args):
    train_loader = build_data_loaders(cfg.DATASETS.TRAINS, transforms=cfg.INPUT.TRANSFORMS_TRAIN, is_train=True, distributed=args.distributed,
                                      batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS)
    target_loader = build_data_loaders(cfg.DATASETS.TARGETS, transforms=cfg.INPUT.TRANSFORMS_TRAIN, is_train=True, distributed=args.distributed,
                                       batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=cfg.DATALOADER.NUM_WORKERS)
    test_loaders = build_data_loaders(cfg.DATASETS.TESTS, transforms=cfg.INPUT.TRANSFORMS_TEST, is_train=False,
                                      distributed=args.distributed, num_workers=cfg.DATALOADER.NUM_WORKERS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_detectors(cfg)
    model.to(device)

    # dis_model = Discriminator(cfg)
    # dis_model.to(device)

    model_without_ddp = model
    # dis_model_without_ddp = dis_model
    if args.distributed:
        model = DistributedDataParallel(convert_sync_batchnorm(model), device_ids=[args.gpu], find_unused_parameters=True)

        # dis_model = DistributedDataParallel(dis_model, device_ids=[args.gpu])
        model_without_ddp = model.module
        # dis_model_without_ddp = dis_model.module

    # optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    '''
    for p in model.parameters():
         print(p)
    '''
    #print(cfg.SOLVER.LR)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    # dis_optimizer = torch.optim.Adam([p for p in dis_model.parameters() if p.requires_grad], cfg.SOLVER.LR, betas=(0.9, 0.999), weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    schedulers = [
        torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA),
        # torch.optim.lr_scheduler.MultiStepLR(dis_optimizer, cfg.SOLVER.STEPS, gamma=cfg.SOLVER.GAMMA),
    ]

    current_epoch = -1
    if args.resume:
        print('Loading from {} ...'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if 'current_epoch' in checkpoint and args.use_current_epoch:
            current_epoch = int(checkpoint['current_epoch'])
        # if 'discriminator' in checkpoint:
        #     dis_model_without_ddp.load_state_dict(checkpoint['discriminator'])

    work_dir = cfg.WORK_DIR
    if args.test_only:
        evaluation(model, test_loaders, device, types=cfg.TEST.EVAL_TYPES, output_dir=work_dir)
        return

    losses_writer = None
    if dist_utils.is_main_process():
        losses_writer = SummaryWriter(os.path.join(work_dir, 'losses'))
        losses_writer.add_text('config', '{}'.format(str(cfg).replace('\n', '  \n')))
        losses_writer.add_text('args', str(args))

    metrics_writers = {}
    if dist_utils.is_main_process():
        test_dataset_names = [loader.dataset.dataset_name for loader in test_loaders]
        for dataset_name in test_dataset_names:
            metrics_writers[dataset_name] = SummaryWriter(os.path.join(work_dir, 'metrics', dataset_name))

    start_time = time.time()
    epochs = cfg.SOLVER.EPOCHS
    global total_steps
    start_epoch = current_epoch + 1
    total_steps = (epochs - start_epoch) * len(train_loader)
    print("Start training, total epochs: {} ({} - {}), total steps: {}".format(epochs - start_epoch, start_epoch, epochs - 1, total_steps))
    for epoch in range(start_epoch, epochs):
        if args.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)
            target_loader.batch_sampler.sampler.set_epoch(epoch)

        def test_func():
            global best_mAP
            updated = False
            metrics = evaluation(model, test_loaders, device, cfg.TEST.EVAL_TYPES, output_dir=work_dir, iteration=global_step)
            if dist_utils.is_main_process() and losses_writer:
                for dataset_name, metric in metrics.items():
                    for k, v in metric.items():
                        metrics_writers[dataset_name].add_scalar('metrics/' + k, v, global_step=global_step)
                        # if k == 'mAP' and v > best_mAP:
                        if k == 'AP' and v > best_mAP:
                            best_mAP = v
                            updated = True
            model.train()

            return updated

        def save_func(filename=None, save_str=None):
            state_dict = {
                'model': model_without_ddp.state_dict(),
                # 'discriminator': dis_model_without_ddp.state_dict(),
                'current_epoch': epoch,
            }
            filename = filename if filename else 'model_epoch_{:02d}.pth'.format(epoch)
            save_path = os.path.join(work_dir, filename)
            dist_utils.save_on_master(state_dict, save_path)
            if dist_utils.is_main_process() and save_str is not None:
                with open(os.path.join(work_dir, 'best.txt'), 'w') as f:
                    f.write(save_str)

            print('Saved to {}'.format(save_path))

        epoch_start = time.time()
        train_one_epoch(model, optimizer, train_loader, target_loader, device, epoch,
                        dis_model=None, dis_optimizer=None,
                        writer=losses_writer, test_func=test_func, save_func=save_func)

        for scheduler in schedulers:
            scheduler.step()

        save_func()

        '''
        if epoch == (epochs - 1):
            test_func()
        '''
        test_func()

        epoch_cost = time.time() - epoch_start
        left = epochs - epoch - 1
        print('Epoch {} ended, cost {}. Left {} epochs, may cost {}'.format(epoch,
                                                                            str(datetime.timedelta(seconds=int(epoch_cost))),
                                                                            left,
                                                                            str(datetime.timedelta(seconds=int(left * epoch_cost)))))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--config-file", help="path to config file", type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", help="Only test the model", action="store_true")
    parser.add_argument("--use_current_epoch", help="if use resume epoch", action="store_true")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    dist_utils.init_distributed_mode(args)

    print(args)
    world_size = dist_utils.get_world_size()
    if world_size != 4:
        lr = cfg.SOLVER.LR * (float(world_size) / 4)
        print('Change lr from {} to {}'.format(cfg.SOLVER.LR, lr))
        cfg.merge_from_list(['SOLVER.LR', lr])

    print(cfg)
    os.makedirs(cfg.WORK_DIR, exist_ok=True)
    if dist_utils.is_main_process():
        with open(os.path.join(cfg.WORK_DIR, 'config.yaml'), 'w') as fid:
            fid.write(str(cfg))
    main(cfg, args)
