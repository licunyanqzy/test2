import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.cluster import SpectralClustering
import numpy as np
import os
from multiprocessing.dummy import Pool as ThreadPool


class Clustering(nn.Module):
    def __init__(self, node_dim, node_embedding_dim, k_group):
        super(Clustering, self).__init__()
        self.node_dim = node_dim
        self.node_embedding_dim = node_embedding_dim
        self.k_group = k_group

        self.node_embedding = nn.Linear(self.node_dim, self.node_embedding_dim)
        self.self_attention = nn.Linear(self.node_embedding_dim * 2, 1)
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=0)

    def cal_edge(self, node):
        n = node.size()[0]      # node [3,192]
        node_other = node.repeat(n, 1).view(n, n, -1)
        node_other_embedding = self.node_embedding(node_other)
        node_self = node_other.transpose(0, 1)
        node_self_embedding = self.node_embedding(node_self)
        node_inter = torch.cat([node_other_embedding, node_self_embedding], dim=-1)
        edge = self.softmax(self.leakyReLU(self.self_attention(node_inter).squeeze(-1)))

        triu_0 = torch.triu(edge, diagonal=0)
        triu_1 = torch.triu(edge, diagonal=1).t()
        edge_output = triu_0 + triu_1       # symmetric ...

        return edge_output

    def kmeans(self, x, ncluster, niter=10):
        N, D = x.size()     # [21,2]
        c = x[torch.randperm(N)[:ncluster]]
        for i in range(niter):
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            c[nanix] = x[torch.randperm(N)[:ndead]]

        group = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
        return group

    def forward(self, node):
        n = node.size()[0]
        edge = self.cal_edge(node)

        # sklearn
        # k_group = int(np.ceil(n/2))     # k_group 改为用 n 表示, 即根据场景人数动态变化
        # sc = SpectralClustering(n_clusters=k_group, affinity="precomputed")
        # sc.fit(edge.detach().cpu().numpy())
        # group = torch.tensor(sc.labels_).cuda().squeeze(0)

        # hand-crafted
        W = edge

        degreeMatrix = torch.sum(W, dim=1)
        L = torch.diag(degreeMatrix) - W
        sqrtDegreeMatrix = torch.diag(1.0 / (degreeMatrix ** 0.5))
        L_sym = torch.matmul(torch.matmul(sqrtDegreeMatrix, L), sqrtDegreeMatrix)
        lam, H = torch.eig(L_sym, eigenvectors=True)    # [21,21]
        lam_r = lam[:, 0]
        t = torch.argsort(lam_r)
        H = torch.cat([H[:, t[0]].unsqueeze(-1), H[:, t[1]].unsqueeze(-1)], dim=1)  # [21,2]
        s = torch.sqrt(torch.sum(H ** 2, dim=1)).unsqueeze(-1)
        H_norm = H / s

        k_group = int(np.ceil(n / 2))
        labels = self.kmeans(H_norm, k_group)
        group = labels

        return group, edge


class SignedGraph(nn.Module):
    def __init__(self, node_dim, node_embedding_dim, k_group):
        super(SignedGraph, self).__init__()
        self.node_dim = node_dim
        self.node_embedding_dim = node_embedding_dim
        self.k_group = k_group

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=0)     # to be continued...

        self.clustering = Clustering(
            node_dim=self.node_dim, node_embedding_dim=self.node_embedding_dim,
            k_group=self.k_group
        )

    def forward(self, node):
        n = node.size()[0]
        group, edge = self.clustering(node)      # tensor[10]

        group_ext = group.unsqueeze(0).repeat(n, 1).view(n, n, 1).squeeze(-1)
        # self_flag = group_ext.diag().repeat(n, 1).view(n, n, 1).squeeze(-1)
        self_flag = group_ext.transpose(1, 0)

        # positive = torch.tensor(self_flag == group_ext, dtype=torch.float).cuda()
        # negative = torch.tensor(self_flag != group_ext, dtype=torch.float).cuda()
        positive = (self_flag == group_ext).float().clone()
        negative = (self_flag != group_ext).float().clone()

        neg_inf = (-1e8) * torch.ones(n, n).cuda()
        positive = torch.where(positive == 0, neg_inf, positive)
        negative = torch.where(negative == 0, neg_inf, negative)

        pos_edge = edge * positive
        neg_edge = edge * negative

        pos_alpha = self.softmax(self.leaky_relu(pos_edge))
        neg_alpha = self.softmax(self.leaky_relu(neg_edge))

        node_ext = node.repeat(n, 1).view(n, n, -1)
        pos_feature = F.relu(pos_alpha.unsqueeze(-1) * node)
        neg_feature = F.relu(neg_alpha.unsqueeze(-1) * node)
        output_feature = torch.sum((pos_feature + neg_feature), dim=1)

        return output_feature


# class Crowd(nn.Module):
#     def __init__(self, ):
#         super(Crowd, self).__init__()
#
#     def forward(self, *input):
#
#         return


class TrajectoryPrediction(nn.Module):
    def __init__(
            self, obs_len, pred_len, noise_dim,
            encoder_input_dim, encoder_hidden_dim,
            graph_node_embedding_dim, k_group,
    ):
        super(TrajectoryPrediction, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.noise_dim = noise_dim
        self.relu = nn.ReLU()

        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_input_dim = encoder_input_dim
        self.inputEmbedding = nn.Linear(2, self.encoder_input_dim)
        self.encoderLSTM = nn.LSTMCell(self.encoder_input_dim, self.encoder_hidden_dim)

        self.decoder_input_dim = encoder_input_dim
        self.decoder_hidden_dim = self.encoder_hidden_dim + self.noise_dim[0]
        self.decoderLSTM = nn.LSTMCell(self.decoder_input_dim, self.decoder_hidden_dim)
        self.hidden2traj = nn.Linear(self.decoder_hidden_dim, 2)

        self.signedGraph = SignedGraph(
            node_dim=self.decoder_hidden_dim, node_embedding_dim=graph_node_embedding_dim, k_group=k_group
        )

    def encoder(self, input_traj):
        batch = input_traj.shape[1]
        encoder_hidden_state = torch.randn(batch, self.encoder_hidden_dim).cuda()
        encoder_cell_state = torch.randn(batch, self.encoder_hidden_dim).cuda()

        traj_embedding = self.relu(self.inputEmbedding(input_traj.contiguous().view(-1, 2)))
        traj_embedding = traj_embedding.view(-1, batch, self.encoder_input_dim)

        for i, input_data in enumerate(
            traj_embedding[: self.obs_len].chunk(
                traj_embedding[: self.obs_len].size(0), dim=0
            )
        ):
            encoder_hidden_state, encoder_cell_state = self.encoderLSTM(
                input_data.squeeze(0), (encoder_hidden_state, encoder_cell_state)
            )

        return encoder_hidden_state

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim
        z_decoder = torch.randn(*noise_shape).cuda()    # 有简化 noise_type == gaussian
        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end - start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h

    def decoder(
            self, input_hidden_state, traj_real, seq_start_end, teacher_forcing_ratio,
    ):
        decoder_hidden_state = input_hidden_state
        decoder_cell_state = torch.zeros_like(decoder_hidden_state).cuda()

        pred_seq = []
        traj_output = traj_real[self.obs_len - 1]

        batch = traj_real.shape[1]
        traj_real_embedding = self.relu(self.inputEmbedding(traj_real.contiguous().view(-1, 2)))
        traj_real_embedding = traj_real_embedding.view(-1, batch, self.decoder_input_dim)

        if self.training:
            for i in range(self.pred_len):
                if random.random() < teacher_forcing_ratio:
                    input_data = traj_real_embedding[-self.pred_len + i]
                else:
                    input_data = self.relu(self.inputEmbedding(traj_output))

                if i % 12 == 4:
                    decoder_hidden_state = self.social(decoder_hidden_state, seq_start_end)

                decoder_hidden_state, decoder_cell_state = self.decoderLSTM(
                    input_data.squeeze(0), (decoder_hidden_state, decoder_cell_state)
                )
                traj_output = self.hidden2traj(decoder_hidden_state)

                pred_seq += [traj_output]
        else:
            for i in range(self.pred_len):
                input_data = self.relu(self.inputEmbedding(traj_output))

                if i % 12 == 4:
                    decoder_hidden_state = self.social(decoder_hidden_state, seq_start_end)

                decoder_hidden_state, decoder_cell_state = self.decoderLSTM(
                    input_data.squeeze(0), (decoder_hidden_state, decoder_cell_state)
                )
                traj_output = self.hidden2traj(decoder_hidden_state)

                pred_seq += [traj_output]

        pred_seq_output = torch.stack(pred_seq)
        return pred_seq_output

    def social(self, hidden_state, seq_start_end):
        # input_list = []
        # for start, end in seq_start_end.data:
        #     input_list.append(hidden_state[start:end, :])
        #
        # pool = ThreadPool()
        # outputs = pool.map(self.signedGraph, input_list)
        # pool.close()
        # pool.join()
        # output_hidden_state = torch.cat(outputs, dim=0)
        # return output_hidden_state

        outputs = []
        for start, end in seq_start_end.data:
            curr_hidden_state = hidden_state[start:end, :]
            curr_graph_output = self.signedGraph(curr_hidden_state)
            outputs.append(curr_graph_output)
        output_hidden_state = torch.cat(outputs, dim=0)
        return output_hidden_state

    def forward(
            self, input_traj, seq_start_end, teacher_forcing_ratio=0.5,
    ):
        encoder_hidden_state = self.encoder(input_traj)
        encoder_hidden_state_noise = self.add_noise(encoder_hidden_state, seq_start_end)
        pred_seq = self.decoder(
            encoder_hidden_state_noise, input_traj, seq_start_end, teacher_forcing_ratio,
        )
        return pred_seq

