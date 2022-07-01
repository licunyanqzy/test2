import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import random
import logging

from data.loader import data_loader
from model import TrajectoryPrediction
import utils
from utils import int_tuple


parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", default="./", help="Directory containing logging file")
parser.add_argument("--dataset_name", default="eth", type=str)
parser.add_argument("--delim", default="\t")
parser.add_argument("--loader_num_workers", default=1, type=int)
parser.add_argument("--skip", default=1, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--seed", default=72, type=int)

parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)

parser.add_argument("--encoder_hidden_dim", default=128, type=int)
parser.add_argument("--encoder_input_dim", default=64, type=int)
parser.add_argument("--graph_node_embedding_dim", default=128, type=int)
parser.add_argument("--k_group", default=3, type=int)

parser.add_argument("--dropout", default=0, type=float)     # dropout rate
parser.add_argument("--alpha", default=0.2, type=float)   # alpha for the leaky relu

# parser.add_argument("--noise_dim", default=64, type=int)
parser.add_argument("--noise_dim", default=(64,), type=int_tuple)
parser.add_argument("--noise_type", default="gaussian", type=str)

parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--num_epoch", default=400, type=int)
parser.add_argument("--best_k", default=20, type=int)

parser.add_argument("--gpu_num", default="0", type=str)
parser.add_argument("--use_gpu", default=True, type=bool)
parser.add_argument("--print_every", default=10, type=int)

parser.add_argument("--resume", default="", type=str)

bestADE = 100


def train(args, model, train_loader, optimizer, epoch, writer):     # , writer
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            seq_start_end,
        ) = batch

        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len:]

        traj_gt = torch.cat((obs_traj, pred_traj_gt), dim=0)
        traj_gt_rel = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

        ratio = np.exp(-(epoch)/20)

        pred_seq_fake = model(traj_gt, seq_start_end, teacher_forcing_ratio=ratio)

        l2_loss = utils.l2_loss(pred_seq_fake, pred_traj_gt_rel, loss_mask, mode="raw").unsqueeze(1)

        l2_loss_sum = utils.l2_loss_sum(l2_loss, seq_start_end, pred_seq_fake.shape[0])

        loss += l2_loss_sum
        losses.update(loss.item(), obs_traj.shape[1])
        loss.backward()
        optimizer.step()

        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)

    writer.add_scalar("train_loss", losses.avg, epoch)


def validate(args, model, val_loader, epoch, writer):
    ADE = utils.AverageMeter("ADE", ":.6f")
    FDE = utils.AverageMeter("FDE", ":.6f")
    progress = utils.ProgressMeter(len(val_loader), [ADE, FDE], prefix="Test: ")

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = [tensor.cuda() for tensor in batch]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            loss_mask = loss_mask[:, args.obs_len:]
            traj_gt = torch.cat((obs_traj, pred_traj_gt), dim=0)

            pred_traj_fake_rel = model(obs_traj, seq_start_end)

            pred_traj_fake_abs = utils.relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            ADE_, FDE_ = utils.cal_ADE_FDE(pred_traj_gt, pred_traj_fake_abs)
            ADE_ = ADE_ / (obs_traj.shape[1] * args.pred_len)
            FDE_ = FDE_ / (obs_traj.shape[1])
            ADE.update(ADE_, obs_traj.shape[1])
            FDE.update(FDE_, obs_traj.shape[1])

            if i % args.print_every == 0:
                progress.display(i)

        logging.info(
            "* ADE {ade.avg:.3f} FDE {fde.avg:.3f}".format(ade=ADE, fde=FDE)
        )
        writer.add_scalar("val_ade", ADE.avg, epoch)

    return ADE.avg


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = utils.get_dset_path(args.dataset_name, "train")
    val_path = utils.get_dset_path(args.dataset_name, "test")

    logging.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logging.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    writer = SummaryWriter()

    model = TrajectoryPrediction(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        noise_dim=args.noise_dim,
        encoder_input_dim=args.encoder_input_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        graph_node_embedding_dim=args.graph_node_embedding_dim,
        k_group=args.k_group
    )
    model.cuda()
    params = ([p for p in model.parameters()])
    optimizer = optim.Adam(params, lr=args.lr)

    if args.resume:     # start from checkpoint
        if os.path.isfile(args.resume):
            logging.info("Restoring from checkpoint {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found as '{}'".format(args.resume))

    global bestADE

    for epoch in range(args.start_epoch, args.num_epoch):
        # train(args, model, train_loader, optimizer, epoch)

        if epoch < 0:   # 调整 ?
            train(args, model, train_loader, optimizer, epoch, writer)
        else:
            train(args, model, train_loader, optimizer, epoch, writer)

            ADE = validate(args, model, val_loader, epoch, writer)
            is_best = ADE < bestADE
            bestADE = min(ADE, bestADE)

            utils.save_checkpoint(  # if ADE > bestADE, save checkpoint
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_ADE": bestADE,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                f"./checkpoint/checkpoint{epoch}.pth.tar",
            )

    writer.close()


if __name__ == '__main__':
    args = parser.parse_args()
    utils.set_logger(os.path.join(args.log_dir, "train.log"))
    checkpoint_dir = "./checkpoint"
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    main(args)