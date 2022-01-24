import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
import utils.config
import os
# from data import dataloader
from utils.optimizer import build_opti_sche
from PointDA.data.dataloader import ScanNet, ModelNet, ShapeNet, label_to_idx, PointcloudScaleAndTranslate
from PointDA.Models import PointNet, DGCNN, PointTransformer
from utils import pc_utils
from utils.checkpoint import resume_model,save_checkpoint,resume_optimizer
from DefRec_and_PCM import DefRec, PCM
from pathlib import Path
import time
from utils.AverageMeter import AverageMeter
from pointnet2_ops import pointnet2_utils
NWORKERS=4
MAX_LOSS = 9 * (10**9)

train_transforms = transforms.Compose(
    [
        PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        PointcloudScaleAndTranslate(),
    ]
)
def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
# parser.add_argument('--exp_name', type=str, default='DefRec_PCM',  help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--dataroot', type=str, default='./data', metavar='N', help='data path')
parser.add_argument('--src_dataset', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--epochs', type=int, default=300, help='number of episode to train')
parser.add_argument('--model', type=str, default='ptrans', choices=['ptrans','pointnet', 'dgcnn'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_voxels', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--DefRec_on_src', type=str2bool, default=False, help='Using DefRec in source')
parser.add_argument('--DefRec_on_target', type=str2bool, default=False, help='Using DefRec in target')
parser.add_argument('--apply_PCM', type=str2bool, default=False, help='Using mixup in source')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--sch_initial_epochs', type=int, default=None, help='number of episode of initsheduler')
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

parser.add_argument('--resume', type=str2bool, default=False, help = 'autoresume training (interrupted by accident)')
parser.add_argument('--config',type = str,help = 'yaml config file')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
args = parser.parse_args()
args.exp_name = args.src_dataset + '_to_' + args.trgt_dataset
## config
config = utils.config.get_config(args)
args.log_name = Path(args.config).stem

if args.epochs != config.max_epoch:
    config.max_epoch = args.epochs
    config.scheduler.kwargs.epochs = args.epochs
print('Src data will train for %d'%(config.max_epoch))
if args.sch_initial_epochs is not None:
    config.scheduler.kwargs.initial_epochs = args.sch_initial_epochs
print('sch_initial_epochs is %d'%(config.scheduler.kwargs.initial_epochs))
# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

random.seed(1)
np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')


# ==================
# Writter
train_writer = None
args.tfboard_path = os.path.join(io.path, 'TFBoard')
if not os.path.exists(args.tfboard_path):
    os.makedirs(args.tfboard_path)
    print('Create TFBoard path successfully at %s' % args.tfboard_path)
if args.local_rank == 0:
    train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
    val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
# ==================
# Read Data
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

src_dataset = args.src_dataset
trgt_dataset = args.trgt_dataset
data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

src_trainset = data_func[src_dataset](io, args.dataroot, 'train')
trgt_trainset = data_func[trgt_dataset](io, args.dataroot, 'train')
trgt_testset = data_func[trgt_dataset](io, args.dataroot, 'test')

# Creating data indices for training and validation splits:
src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
trgt_train_sampler, trgt_valid_sampler = split_set(trgt_trainset, trgt_dataset, "target")

# dataloaders for source and target
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                               sampler=src_train_sampler, drop_last=True)
src_val_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                             sampler=src_valid_sampler)
trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                                sampler=trgt_train_sampler, drop_last=True)
trgt_val_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                                  sampler=trgt_valid_sampler)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)

# ==================
# Init Model
# ==================
if args.model == 'pointnet':
    model = PointNet(args)
elif args.model == 'dgcnn':
    model = DGCNN(args)
elif args.model == 'ptrans':
    model = PointTransformer(config.model)
else:
    raise Exception("Not implemented")

# n_eles = 0
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name,param.size())
#         n_eles=n_eles+param.numel()
# print('total params %d'%(n_eles))
##
# metrics
src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
src_best_val_loss = trgt_best_val_loss = MAX_LOSS
start_epoch = 0
criterion = nn.CrossEntropyLoss() 

# lookup table of regions means
lookup = torch.Tensor(pc_utils.region_mean(args.num_regions)).to(device)

# best_model = io.save_model(model)

if args.resume:
    start_epoch, src_best_val_acc = resume_model(model,args,io)
    best_model = copy.deepcopy(model)
    best_val_epoch, src_best_val_acc = resume_model(best_model,args,io,'ckpt-best.pth')
else:
    if args.ckpts is not None:
        model.load_model_from_ckpt(args.ckpts)
    else:
        io.cprint('Training from scratch')

if args.cuda:
    # model = model.to(device)
    model = model.to(args.local_rank)
# Handle multi-gpu
if (device.type == 'cuda'):
    model = nn.DataParallel(model).cuda()
if not args.resume:
    best_model = copy.deepcopy(model)

if False:
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" \
        else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(opt, args.epochs)
else:
    opt, scheduler = build_opti_sche(model, config)

if args.resume:
    resume_optimizer(opt, args, io)
# ==================
# Validation/test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0, writter=None):

    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0
    losses = AverageMeter(['loss', 'acc'])
    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            batch_size = data.size()[0]
            ret = model(data)
            loss, acc_batch = model.module.get_loss_acc(ret, labels)
            print_losses['cls'] += loss.item() * batch_size
            losses.update([loss.item(), acc_batch.item()])

            # evaluation metrics
            preds = ret.max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    if writter is not None:
        outstr = "%s-%s" % (set_type,partition)
        writter.add_scalar('%s/Epoch/Cls_loss'%(outstr), losses.avg(0), epoch)
        writter.add_scalar('%s/Epoch/Cls_acc'%(outstr), losses.avg(1), epoch)

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    losses.reset()

    return test_acc, print_losses['cls'], conf_mat


# ==================
# Train
# ==================
model.zero_grad()

for epoch in range(start_epoch,args.epochs):
    model.train()
    epoch_start_time = time.time()
    batch_start_time = time.time()
    num_iter = 0
    # init data structures for saving epoch stats
    
    src_print_losses = {"total": 0.0}
    if args.apply_PCM:
        src_print_losses['mixup']=0.0
        Cls_losses = AverageMeter(['mixuploss'])
    else:
        src_print_losses['cls']=0.0
        Cls_losses = AverageMeter(['clsloss', 'acc'])
    batch_time = AverageMeter()
    data_time = AverageMeter()
    if args.DefRec_on_src:
        src_print_losses['DefRec'] = 0.0
    if args.DefRec_on_target:
        trgt_print_losses = {'DefRec': 0.0}
        Trgt_defrecloss = AverageMeter(['trgt_defrecloss'])
    src_count = trgt_count = 0.0
    npoints = config.npoints
    n_batches = len(src_train_loader)
    idx = 0
    for data1, data2 in zip(src_train_loader, trgt_train_loader):
    # for idx, data1 in enumerate(src_train_loader):
        num_iter +=1
        n_itr = epoch * n_batches + idx

        opt.zero_grad()
        data_time.update(time.time() - batch_start_time)

        if npoints == 1024:
            point_all = 1200
        elif npoints == 4096:
            point_all = 4800
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        #### source data ####
        if data1 is not None:
            src_data, src_label = data1[0].to(device), data1[1].to(device).squeeze()
            # print(src_data.shape)
            batch_size = src_data.size()[0]
            if src_data.size(1) < point_all:
                point_all = src_data.size(1)
            # if False:
                fps_idx = pointnet2_utils.furthest_point_sample(src_data, point_all)  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                src_data = pointnet2_utils.gather_operation(src_data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
                # import pdb; pdb.set_trace()
                src_data = train_transforms(src_data)
            if args.apply_PCM:
                src_data_pcm = src_data.clone()
                src_data_pcm = src_data_pcm.permute(0, 2, 1)
                src_data_pcm, mixup_vals = PCM.mix_shapes(args, src_data_pcm, src_label)
                src_data_pcm = src_data_pcm.permute(0, 2, 1).contiguous()
                # print('src_data_pcm',src_data_pcm.shape)
                src_cls_logits = model(src_data_pcm)
                loss = PCM.calc_loss_ptrans(args, src_cls_logits, mixup_vals, criterion)
                _loss = loss
                _loss.backward()
                src_print_losses['mixup'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                Cls_losses.update([loss.item()])

                if train_writer is not None:
                    train_writer.add_scalar('Src-Train/Batch/Mixup_loss', loss.item(), n_itr)
 
            else:
                ret = model(src_data)
                loss, scr_acc_batch = model.module.get_loss_acc(ret, src_label)
                # loss *= (1 - args.DefRec_weight)
                _loss = loss
                _loss.backward()
                src_print_losses['cls'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size

                Cls_losses.update([loss.item(), scr_acc_batch.item()])
                if train_writer is not None:
                    train_writer.add_scalar('Src-Train/Batch/Cls_loss', loss.item(), n_itr)
                    train_writer.add_scalar('Src-Train/Batch/Cls_Acc', scr_acc_batch.item(), n_itr)
            src_count += batch_size

        if data2 is not None:
            if args.DefRec_on_target:
                trgt_data, trgt_label = data2[0].to(device), data2[1].to(device).squeeze()
                trgt_data = trgt_data.permute(0, 2, 1)
                batch_size = trgt_data.size()[0]
                trgt_data_orig = trgt_data.clone()
                device = torch.device("cuda:" + str(trgt_data.get_device()) if args.cuda else "cpu")

                trgt_data, trgt_mask = DefRec.deform_input(trgt_data, lookup, args.DefRec_dist, device)
                trgt_logits = model(trgt_data.permute(0,2,1).contiguous(), activate_DefRec=True)
                loss = DefRec.calc_loss_ptrans(args, trgt_logits, trgt_data_orig, trgt_mask)
                loss.backward()

                trgt_print_losses['DefRec'] += loss.item() * batch_size
                Trgt_defrecloss.update([loss.item()])

                trgt_count += batch_size
                if train_writer is not None:
                    train_writer.add_scalar('Target-Train/Batch/Defrec_loss', loss.item(), n_itr)

            

        # forward
        if num_iter == config.step_per_update:
            if config.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip, norm_type=2)
            num_iter = 0
            opt.step()
            model.zero_grad()
        
        

        if train_writer is not None:
            train_writer.add_scalar('Scr-Train/Batch/LR', opt.param_groups[0]['lr'], n_itr)
        
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()

        if idx %20 ==0:
            io.cprint('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %s lr = %.6f' % \
                (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(), \
                    Cls_losses.avg(0), opt.param_groups[0]['lr']))
            if not args.apply_PCM:
                io.cprint('Acc = %s' % (Cls_losses.avg(1)))
            if args.DefRec_on_target:
                io.cprint('Trgt Defrec loss = %s'%(Trgt_defrecloss.avg(0)))
        idx += 1
        
        

    if isinstance(scheduler, list):
        for item in scheduler:
            item.step(epoch)
    else:
        scheduler.step(epoch)

    epoch_end_time = time.time()
    
    if train_writer is not None:
        train_writer.add_scalar('Src-Train/Epoch/Loss', Cls_losses.avg(0), epoch)
        if not args.apply_PCM:
            train_writer.add_scalar('Src-Train/Epoch/Acc', Cls_losses.avg(1), epoch)

    print('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' % \
        (epoch,  epoch_end_time - epoch_start_time, Cls_losses.avg(0)))
    if not args.apply_PCM:
        print('Acc = %s'%Cls_losses.avg(1))

    # print progress
    src_print_losses = {k: v * 1.0 / src_count for (k, v) in src_print_losses.items()}
    src_acc = io.print_progress("Source", "Trn", epoch, src_print_losses)
    trgt_print_losses = {k: v * 1.0 / trgt_count for (k, v) in trgt_print_losses.items()}
    trgt_acc = io.print_progress("Target", "Trn", epoch, trgt_print_losses)

    #===================
    # Validation
    #===================
    src_val_acc, src_val_loss, src_conf_mat = test(src_val_loader, model, "Source", "Val", epoch, writter = train_writer)
    trgt_val_acc, trgt_val_loss, trgt_conf_mat = test(trgt_val_loader, model, "Target", "Val", epoch,writter = train_writer)

    # save model according to best source model (since we don't have target labels)
    if src_val_acc > src_best_val_acc:
        src_best_val_acc = src_val_acc
        src_best_val_loss = src_val_loss
        trgt_best_val_acc = trgt_val_acc
        trgt_best_val_loss = trgt_val_loss
        best_val_epoch = epoch
        best_epoch_conf_mat = trgt_conf_mat
        best_model = io.save_model(model)
        save_checkpoint(model, opt, epoch, src_val_acc, src_best_val_acc, 'ckpt-best', args, io)
    save_checkpoint(model, opt, epoch, src_val_acc, src_best_val_acc, 'ckpt-last', args, io)
    if args.epochs-epoch<5:
        save_checkpoint(model, opt, epoch, src_val_acc, src_best_val_acc, f'ckpt-epoch-{epoch:03d}', args, io)

if train_writer is not None:
        train_writer.close()
io.cprint("Best model was found at epoch %d, source validation accuracy: %.4f, source validation loss: %.4f,"
          "target validation accuracy: %.4f, target validation loss: %.4f"
          % (best_val_epoch, src_best_val_acc, src_best_val_loss, trgt_best_val_acc, trgt_best_val_loss))
io.cprint("Best validtion model confusion matrix:")
io.cprint('\n' + str(best_epoch_conf_mat))

#===================
# Test
#===================
model = best_model
trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, model, "Target", "Test", 0)
io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_best_val_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))
