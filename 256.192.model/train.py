import os
import sys
import argparse
import time
from progress.bar import Bar

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

this_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(this_dir, '..')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)


from config import cfg
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from networks import network
from dataloader.mscocoMulti import MscocoMulti


def main(args):
    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr*args.num_gpus,
                                 weight_decay=cfg.weight_decay)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    cudnn.enabled = False
    # cudnn.benchmark = True
    print('    Total params: %.2fMB' % (sum(p.numel() for p in model.parameters()) / (1024 * 1024) * 4))

    train_loader = torch.utils.data.DataLoader(
        MscocoMulti(cfg),
        batch_size=cfg.batch_size * args.num_gpus, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, lr)
        print('train_loss: ', train_loss)

        # append logger file
        logger.append([epoch + 1, lr, train_loss])

        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint=args.checkpoint)

    logger.close()


def train(train_loader, model, optimizer, lr):
    # prepare for refine loss
    def ohkm(loss, top_k):
        ohkm_loss = 0.
        for i in range(loss.size(0)):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += tmp_loss.mean()
        ohkm_loss /= loss.size(0)
        return ohkm_loss

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    bar = Bar('Train', max=len(train_loader))

    end = time.time()
    for i, (inputs, targets, valid, meta) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_var = inputs.cuda()

        target15, target11, target9, target7 = targets
        refine_target_var = target7.cuda()
        valid_var = valid.cuda()

        # compute output
        global_outputs, refine_output = model(input_var)

        num_points = target15.size(1)

        loss = None
        global_loss_record = 0.
        # comput global loss and refine loss
        for global_output, label in zip(global_outputs, targets):
            global_label = label * (valid > 1.1).float().view(-1, num_points, 1, 1)
            global_loss = F.mse_loss(global_output, label.cuda()) / 2.
            if loss is None:
                loss = global_loss
            else:
                loss += global_loss
            global_loss_record += global_loss.item()
        refine_loss = F.mse_loss(refine_output, refine_target_var, reduction='none')
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        refine_loss *= (valid_var > 0.1).float().view(-1, num_points)
        refine_loss = ohkm(refine_loss, 8)
        loss += refine_loss
        refine_loss_record = refine_loss.item()

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        bar_format_string = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                            'LR: {lr:.6f} Loss: {loss1:.6f}-{loss2:.6f}-{loss3:.6f}-{loss4:.6f}'

        bar.suffix = bar_format_string.format(batch=i,
                                              size=len(train_loader),
                                              data=data_time.avg,
                                              bt=batch_time.avg,
                                              total=bar.elapsed_td,
                                              eta=bar.eta_td,
                                              lr=lr,
                                              loss1=loss.item(),
                                              loss2=global_loss_record,
                                              loss3=refine_loss_record,
                                              loss4=losses.avg)
        bar.next()
    bar.finish()

    return losses.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=4, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('--epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    main(parser.parse_args())
