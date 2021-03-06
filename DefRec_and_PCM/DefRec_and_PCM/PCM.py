import torch
import numpy as np
import utils.pc_utils as pc_utils


def mix_shapes(args, X, Y):
    """
    combine 2 shapes arbitrarily in each batch.
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    Input:
        X, Y - shape and corresponding labels
    Return:
        mixed shape, labels and proportion
    """
    mixed_X = X.clone()
    batch_size, _, num_points = mixed_X.size()

    # uniform sampling of points from each shape
    device = torch.device("cuda:" + str(X.get_device()) if args.cuda else "cpu")
    batch_size, _, num_points = X.size()
    index = torch.randperm(batch_size).to(device)  # random permutation of examples in batch

    # draw lambda from beta distribution
    lam = np.random.beta(args.mixup_params, args.mixup_params) if args.mixup_params > 0 else 1.0

    num_pts_a = round(lam * num_points)
    num_pts_b = num_points - round(lam * num_points)

    pts_indices_a, pts_vals_a = pc_utils.farthest_point_sample(args, X, num_pts_a)
    pts_indices_b, pts_vals_b = pc_utils.farthest_point_sample(args, X[index, :], num_pts_b)
    mixed_X = torch.cat((pts_vals_a, pts_vals_b), 2)  # convex combination
    points_perm = torch.randperm(num_points).to(device)  # draw random permutation of points in the shape
    mixed_X = mixed_X[:, :, points_perm]

    Y_a = Y.clone()
    Y_b = Y[index].clone()

    return mixed_X, (Y_a, Y_b, lam)

def mix_shapes_segmentation(args, X, Y):
    """
    combine 2 shapes arbitrarily in each batch for segmentation task.
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    Input:
        X, Y - a batch of pc and corresponding labels for each point
    Return:
        mixed shape
    """
    batch_size, _, num_points = X.size()

    # uniform sampling of points from each shape
    device = torch.device("cuda:" + str(X.get_device()) if args.gpus[0] != -1 else "cpu")
    batch_size, _, num_points = X.size()
    index = torch.randperm(batch_size).to(device)  # random permutation of examples in batch

    # draw lambda from beta distribution
    lam = np.random.beta(args.mixup_params, args.mixup_params) if args.mixup_params > 0 else 1.0

    num_pts_a = round(lam * num_points)
    num_pts_b = num_points - round(lam * num_points)

    pts_indices_a, pts_vals_a = pc_utils.farthest_point_sample(args, X, num_pts_a)
    pts_indices_b, pts_vals_b = pc_utils.farthest_point_sample(args, X[index, :], num_pts_b)
    mixed_X = torch.cat((pts_vals_a, pts_vals_b), 2)  # convex combination
    Y_a = Y.gather(1, pts_indices_a)  # take relevant point labels for a
    Y_b = Y[index, :].gather(1, pts_indices_b)  # take relevant point labels from b
    mixed_Y = torch.cat((Y_a, Y_b), 1)

    points_perm = torch.randperm(num_points).to(device)  # draw random permutation of points in the shape
    mixed_X = mixed_X[:, :, points_perm]
    mixed_Y = mixed_Y[:, points_perm]

    return mixed_X, mixed_Y


def calc_loss(args, logits, mixup_vals, criterion):
    """
    Calculate loss between 2 shapes
    Input:
        logits
        mixup_vals: label of first shape, label of second shape and mixing coefficient
        criterion: loss function
    Return:
        loss
    """
    Y_a, Y_b, lam = mixup_vals
    loss = lam * criterion(logits['cls'], Y_a) + (1 - lam) * criterion(logits['cls'], Y_b)
    loss *= (1 - args.DefRec_weight)
    return loss

def calc_loss_ptrans(args, logits, mixup_vals, criterion):
    """
    Calculate loss between 2 shapes
    Input:
        logits
        mixup_vals: label of first shape, label of second shape and mixing coefficient
        criterion: loss function
    Return:
        loss
    """
    Y_a, Y_b, lam = mixup_vals
    loss = lam * criterion(logits, Y_a) + (1 - lam) * criterion(logits, Y_b)
    # loss *= (1 - args.DefRec_weight)
    return loss