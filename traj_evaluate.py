import argparse
import os
import torch

from data.loader import data_loader
from model import TrajectoryPrediction
import utils


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
parser.add_argument("--k_group", default=5, type=int)

parser.add_argument("--dropout", default=0, type=float)     # dropout rate
parser.add_argument("--alpha", default=0.2, type=float)   # alpha for the leaky relu

parser.add_argument("--noise_dim", default=64, type=int)
parser.add_argument("--noise_type", default="gaussian", type=str)

parser.add_argument("--best_k", default=20, type=int)

parser.add_argument("--gpu_num", default="0", type=str)
parser.add_argument("--dset_type", default="test", type=str)
parser.add_argument(
    "--resume", default="./checkpoint/checkpoint10.pth.tar", type=str,
    metavar="PATH", help="path to latest checkpoint (default: none)"
)
parser.add_argument("--num_samples", default=20, type=int)


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def get_model(checkpoint):
    n_units = [
        [args.action_encoder_hidden_dim]
        + [int(x) for x in args.hidden_units.strip().split(",")]
        + [args.action_encoder_hidden_dim]
    ]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    model = TrajectoryPrediction(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        noise_dim=args.noise_dim,
        encoder_input_dim=args.encoder_input_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        graph_node_embedding_dim=args.graph_node_embedding_dim,
        k_group=args.k_group
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()
    return model


def main(args):
    checkpoint = torch.load(args.resume)
    model = get_model(checkpoint)
    path = utils.get_dset_path(args.dataset_name, args.dset_type)
    _, loader = data_loader(args, path)
    ADE, FDE = evaluate(args, loader, model)
    print(
        "Dataset: {}, Pred Len: {}, ADE: {:.12f}, FDE: {:.12f}".format(
            args.dataset_name, args.pred_len, ADE, FDE
        )
    )


def evaluate(args, loader, model):
    ADE_outer, FDE_outer = [], []
    traj_sum = 0

    with torch.no_grad():
        for batch in loader:
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

            ADE, FDE = [], []
            traj_sum += pred_traj_gt.size(1)

            for _ in range(args.num_samples):
                # traj_rel_gt = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)

                pred_traj_fake_rel = model(obs_traj, obs_traj_rel, seq_start_end)

                # pred_traj_fake = pred_action_fake
                # pred_traj_fake_predpart = pred_traj_fake[-args.pred_len:]
                pred_traj_fake_abs = utils.relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                ADE_, FDE_ = utils.cal_ADE_FDE(pred_traj_gt, pred_traj_fake_abs, mode="raw")

                temp = utils.loss_test(pred_traj_fake_abs, pred_traj_gt)
                print(temp)

                ADE.append(ADE_)
                FDE.append(FDE_)

            ADE_sum = evaluate_helper(ADE, seq_start_end)
            FDE_sum = evaluate_helper(FDE, seq_start_end)
            ADE_outer.append(ADE_sum)
            FDE_outer.append(FDE_sum)

        ADE_output = sum(ADE_outer) / (traj_sum * args.pred_len)
        FDE_output = sum(FDE_outer) / traj_sum

        return ADE_output, FDE_output


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(72)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    main(args)