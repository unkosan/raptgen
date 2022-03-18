#!/usr/bin/env python
# coding: utf-8

# todo: このファイルでは pHMM に関するアルゴリズムをまとめる。weblogo に関するアルゴリズムもここに集約する。

import math
from typing import Callable, List, Tuple, Union
import torch.nn.functional as F
from torch import Tensor, nn
import torch
import numpy as np
import pandas as pd

from raptgen.core.preprocessing import Transition, NucleotideID, State, ID_encode

def kld_loss(mus: Tensor, logvars: Tensor):
    """KL Divergence 項（正則化項）の計算を行う。
    
    Parameters
    ----------
    mus : Tensor
        エンコーダで潜在空間上に投影されるガウス分布の平均ベクトルのバッチセット。
        (`batch`, `embed_dim`) の 2 次元構成である必要がある。
    logvars: Tensor
        エンコーダで潜在空間上に投影されるガウス分布の分散ベクトルのバッチセット。
        (`batch`, `embed_dim`) の 2 次元構成である必要がある。
        
    Returns
    -------
    kl_divergences : Tensor
        バッチの各要素毎で KL 距離を計算した値のセット。
        (`batch`, ) の一次元構成である。
    """
    kl_divergence = - 0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp(), dim=1)
    return kl_divergence

def profile_hmm_vae_loss(
    batch_input: Tensor, 
    transition_probs: Tensor,
    emission_probs: Tensor,
    mus: Tensor, 
    logvars: Tensor, 
    beta: float = 1, 
    force_matching: bool = False, 
    match_cost: float = 5,
    split_ce_kld: bool = False,
    ) -> Tensor:
    """pHMM デコーダを使用する際の損失関数。

    Parameters
    ----------
    batch_input : Tensor
        VAE モデルに入力したバッチの値。(`batch`, `string_length`) の 2 次元構成になっている必要があり，各塩基は `NucleotideID` に示された値で表現されていなければならない。パディングも許可される。
    transition_probs : Tensor
        pHMM の遷移確率。(`batch`, `model_length`+1, `combi of 'from' and 'to'`) の 3 次元構成になっており，全確率は対数で表現されていることとする。
    emission_probs : Tensor
        pHMM の出力確率。(`batch`, `model_length`, `nucleotide_type`=4) の 3 次元構成になっている必要があり，全確率は対数で表現されていることとする。
    mus : Tensor
        エンコーダで潜在空間上に投影されるガウス分布の平均ベクトルのバッチセット。
        (`batch`, `embed_dim`) の 2 次元構成である必要がある。
    logvars : Tensor
        エンコーダで潜在空間上に投影されるガウス分布の分散ベクトルのバッチセット。
        (`batch`, `embed_dim`) の 2 次元構成である必要がある。
    beta : float, default = 1
        再構成誤差項に対する正則化項の影響度の割合。
        例えば `beta = 2` であれば損失関数は 再構成誤差項 + `beta`×正則化項 ということになる。
    force_matching : bool
        遷移確率において matching to matching への遷移にペナルティを科すかを決定する。`True` のときペナルティを科す。
    match_cost : float, default = 5.0
        Matching の方に傾く度合いを示す。大きいほど matching to matcing への傾向が高くなる。`1` のとき他の `combi of 'from' and 'to'` と重みが等しくなる。
    split_ce_kld : True, default = False
        返却時に再構成誤差項と正則化項を別々に分けたテンソルとして返却するかを指定する。これを指定するとき，`beta = 1` となる。
    
    Returns
    -------
    vae_loss : Tensor
        batch 内の各要素に対する ELBO の平均値が出力される。
        `split_ce_kld` を指定した際は再構成誤差項と正則化項のそれぞれの平均値が結合されたテンソルとして出力される。
    """
    assert(batch_input.dim() == 2)
    assert(transition_probs.dim() == 3)
    assert(emission_probs.dim() == 3)
    assert(mus.dim() == 2)
    assert(logvars.dim() == 2)

    reconstruction_error = profile_hmm_loss(
        transition_probs = transition_probs, 
        emission_probs = emission_probs,
        batch_input = batch_input,
    ).mean()

    if force_matching == True:
        reconstruction_error += force_matching_loss(
            transition_probs = transition_probs,
            match_cost = match_cost,
        ).mean()

    regularization_error = kld_loss(mus, logvars).mean()

    if split_ce_kld == True:
        return Tensor([reconstruction_error, regularization_error])
    else:
        return reconstruction_error + beta * regularization_error

def profile_hmm_loss_version_sequence(
    transition_probs: Tensor, 
    emission_probs: Tensor,
    batch_input: Tensor, 
    ) -> Tensor:
    """pHMM の遷移確率と出力確率をとり，`batch_input` で示された元の配列が生成される確率を計算する。
    
    Parameters
    ----------
    transition_probs : Tensor
        pHMM の遷移確率。(`batch`, `model_length`+1, `combi of 'from' and 'to'`) の 3 次元構成になっており，全確率は対数で表現されていることとする。
    emission_probs : Tensor
        pHMM の出力確率。(`batch`, `model_length`, `nucleotide_type`=4) の 3 次元構成になっている必要があり，全確率は対数で表現されていることとする。
    batch_input : Tensor
        VAE モデルに入力したバッチの値。(`batch`, `string_length`) の 2 次元構成になっている必要があり，各塩基は `NucleotideID` に示された値で表現されていなければならない。パディングも許可される。
    
    Returns
    -------
    probs : Tensor
        遷移確率，出力確率が決まっていた際に `batch_input` が生成された確率。
        (`batch`,) の 1 次元構成になっている。
    """
    batch_size = batch_input.shape[0]

    result_list = list()

    for batch_elem in range(batch_size):
        a = transition_probs[batch_elem] # (state_from, state_to, motif_len + 1)
        e_m = emission_probs[batch_elem] # (motif_len, nucleotype_AUCG)
        motif_len = e_m.shape[1]
        input_seq = batch_input[batch_elem] # sequence of NucleotideID

        # padding を抜き取る。
        input_seq = torch.masked_select(input_seq, input_seq.ne(NucleotideID.PAD))
        random_len = len(input_seq)

        # forward アルゴリズムを対数形式で行う。
        # (match or insert or delete, motif_len, random_len) の三次元の動的計画表を用意する。
        f = torch.ones(
            (3, motif_len + 1, random_len + 1),
            device = batch_input.device
        ) * (-100)
        f[State.M, 0, 0] = 0

        for l in range(random_len + 1): # locus starts from '1'
            for k in range(motif_len + 1): # motif starts from '1' but model starts from '0'
                # for state M
                if l != 0 and k != 0:
                    f[State.M, k, l] \
                        = e_m[k-1, input_seq[l-1]] \
                        + torch.logsumexp(torch.stack((
                            a[k-1, Transition.M2M] + f[State.M, k-1, l-1],
                            a[k-1, Transition.I2M] + f[State.I, k-1, l-1],
                            a[k-1, Transition.D2M] + f[State.D, k-1, l-1],
                        )), dim=0)
                # for state I
                if l != 0:
                    f[State.I, k, l] \
                        = math.log(1/4) \
                        + torch.logsumexp(torch.stack((
                            a[k, Transition.M2I] + f[State.M, k, l-1],
                            a[k, Transition.I2I] + f[State.I, k, l-1],
                        )), dim=0)
                # for state D
                if k != 0:
                    f[State.D, k, l] \
                        = torch.logsumexp(torch.stack((
                            a[k-1, Transition.M2D] + f[State.M, k-1, l],
                            a[k-1, Transition.D2D] + f[State.D, k-1, l],
                        )), dim=0)
        
        f[State.M, motif_len, random_len] += a[motif_len, Transition.M2M]
        f[State.I, motif_len, random_len] += a[motif_len, Transition.I2M]
        f[State.D, motif_len, random_len] += a[motif_len, Transition.D2M]

        result_list.append(-torch.logsumexp(f[:, motif_len, random_len], dim=0))

    return torch.stack(result_list)

def profile_hmm_loss(
    transition_probs: Tensor, 
    emission_probs: Tensor,
    batch_input: Tensor, 
    ) -> Tensor:
    """pHMM の遷移確率と出力確率をとり，`batch_input` で示された元の配列が生成される確率を計算する。
    
    Parameters
    ----------
    transition_probs : Tensor
        pHMM の遷移確率。(`batch`, `model_length`+1, `combi of 'from' and 'to'`) の 3 次元構成になっており，全確率は対数で表現されていることとする。
    emission_probs : Tensor
        pHMM の出力確率。(`batch`, `model_length`, `nucleotide_type`=4) の 3 次元構成になっている必要があり，全確率は対数で表現されていることとする。
    batch_input : Tensor
        VAE モデルに入力したバッチの値。(`batch`, `string_length`) の 2 次元構成になっている必要があり，各塩基は `NucleotideID` に示された値で表現されていなければならない。パディングも許可される。
    
    Returns
    -------
    probs : Tensor
        遷移確率，出力確率が決まっていた際に `batch_input` が生成された確率。
        (`batch`,) の 1 次元構成になっている。
    """
    batch_size = batch_input.shape[0]
    motif_len = emission_probs.shape[1]

    # 下準備
    record_list = list()
    for elem_index in range(batch_size):
        input_seq = batch_input[elem_index] # sequence of NucleotideID

        # padding を抜き取る。
        input_seq = torch.masked_select(input_seq, input_seq.ne(NucleotideID.PAD))
        record_list.append({
            'seq': input_seq,
            'len': len(input_seq),
            'a': transition_probs[elem_index],
            'e_m': emission_probs[elem_index],
        })
    batch_pd = pd.DataFrame.from_records(record_list)

    result_list = list()
    pd.set_option('mode.chained_assignment', None)
    for seq_length in np.unique(batch_pd['len']):
        equal_len_pd = batch_pd[batch_pd['len'] == int(seq_length)]
        set_size = len(equal_len_pd)

        input_seq_set = torch.stack(equal_len_pd['seq'].to_list())
        a_set = torch.stack(equal_len_pd['a'].to_list())
        e_m_set = torch.stack(equal_len_pd['e_m'].to_list())

        # forward アルゴリズムを対数形式で行う。
        # (match or insert or delete, motif_len, random_len) の三次元の動的計画表を用意する。
        f_set = torch.ones(
            size = (set_size, 3, motif_len + 1, seq_length + 1),
            device = input_seq_set.device
        ) * (-100)
        # f_set[:, State.M, 0, 0] = 0
        f_set[:, 0, 0, 0] = 0

        for l in range(seq_length + 1): # locus starts from '1'
            for k in range(motif_len + 1): # motif starts from '1' but model starts from '0'
                # for state M
                if l != 0 and k != 0:
                    f_set[:, State.M, k, l] \
                        = e_m_set[:, k-1] \
                            .gather(1, input_seq_set[:, l-1:l])[:, 0] \
                        + torch.logsumexp(torch.stack((
                            a_set[:, k-1, Transition.M2M] + f_set[:, State.M, k-1, l-1],
                            a_set[:, k-1, Transition.I2M] + f_set[:, State.I, k-1, l-1],
                            a_set[:, k-1, Transition.D2M] + f_set[:, State.D, k-1, l-1],
                        )), dim=0)
                # for state I
                if l != 0:
                    f_set[:, State.I, k, l] \
                        = np.log(1/4) \
                        + torch.logsumexp(torch.stack((
                            a_set[:, k, Transition.M2I] + f_set[:, State.M, k, l-1],
                            a_set[:, k, Transition.I2I] + f_set[:, State.I, k, l-1],
                        )), dim=0)
                # for state D
                if k != 0:
                    f_set[:, State.D, k, l] \
                        = torch.logsumexp(torch.stack((
                            a_set[:, k-1, Transition.M2D] + f_set[:, State.M, k-1, l],
                            a_set[:, k-1, Transition.D2D] + f_set[:, State.D, k-1, l],
                        )), dim=0)
        
        f_set[:, State.M, motif_len, seq_length] += a_set[:, motif_len, Transition.M2M]
        f_set[:, State.I, motif_len, seq_length] += a_set[:, motif_len, Transition.I2M]
        f_set[:, State.D, motif_len, seq_length] += a_set[:, motif_len, Transition.D2M]

        val_set = - torch.logsumexp(f_set[:, :, motif_len, seq_length], dim=1)

        result_list += list(zip(
            list(equal_len_pd.index),
            list(val_set),
        ))

    pd.reset_option('mode.chained_assignment')

    result_sorted = sorted(
        result_list, 
        key = lambda x: x[0],
    )
    
    result_tensor = Tensor([
        value_tuple[1]
        for value_tuple in result_sorted
    ])

    return result_tensor

def force_matching_loss(transition_probs: Tensor, match_cost: float = 5.0) -> Tensor:
    """遷移確率において，Match to Match の確率が小さければ小さい程大きな損失値を与える損失関数を定義する。
    これを初期 epoch における損失関数に付け加えることにより，モチーフの学習がうまくいく。論文中では State_transition_loss として定義されている。
    
    Parameters
    ----------
    transition_probs : Tensor
        遷移確率。(`batch`, `model_length`+1, `combi of 'from' and 'to'`) の三次元配列。
    match_cost : float, default = 5.0
        Matching の方に傾く度合いを示す。大きいほど matching to matcing への傾向が高くなる。`1` のとき他の `combi of 'from' and 'to'` と重みが等しくなる。
    
    Returns
    -------
    force_matching_loss : Tensor
        Matching to Matcing の確率が全体的に小さいほど大きくなるような損失関数。
        (`batch`, ) の一次元構成になっている。
    """
    loss \
        = math.log( (match_cost + 1) * match_cost / 2 ) \
        + torch.sum(
            (match_cost - 1) * transition_probs[:, :, Transition.M2M], 
            dim=1,
        )
    return - loss

    # for i in range(random_len + 1):
    #     for j in range(motif_len + 1):
    #         # State M
    #         if j*i != 0:
    #             f[:, State.M, j, i] \
    #                 = e_m[:, j-1].gather(1, batch_input[:, i-1:i])[:, 0] \
    #                 + torch.logsumexp(torch.stack((
    #                     a[:, j - 1, Transition.M2M] +
    #                     f[:, State.M, j - 1, i - 1],
    #                     a[:, j - 1, Transition.I2M] +
    #                     f[:, State.I, j - 1, i - 1],
    #                     a[:, j - 1, Transition.D2M] +
    #                     f[:, State.D, j - 1, i - 1])), dim=0)
                
    #         # State I
    #         if i != 0:
    #             f[:, State.I, j, i] \
    #                 = - 1.3863 \
    #                 + torch.logsumexp(torch.stack((
    #                     a[:, j, Transition.M2I] +
    #                     f[:, State.M, j, i-1],
    #                     # Removed D-to-I transition
    #                     # a[:, j, Transition.D2I] +
    #                     # F[:, State.D, j, i-1],
    #                     a[:, j, Transition.I2I] +
    #                     f[:, State.I, j, i-1]
    #                 )), dim=0)

    #         # State D
    #         if j != 0:
    #             f[:, State.D, j, i] = \
    #                 torch.logsumexp(torch.stack((
    #                     a[:, j - 1, Transition.M2D] +
    #                     f[:, State.M, j - 1, i],
    #                     # REMOVED I-to-D transition
    #                     # a[:, j - 1, Transition.I2D] +
    #                     # F[:, State.I, j - 1, i],
    #                     a[:, j - 1, Transition.D2D] +
    #                     f[:, State.D, j - 1, i]
    #                 )), dim=0)

    # # final I->M transition
    # f[:, State.M, motif_len, random_len] += a[:,
    #                                           motif_len, Transition.M2M]
    # f[:, State.I, motif_len, random_len] += a[:,
    #                                           motif_len, Transition.I2M]
    # f[:, State.D, motif_len, random_len] += a[:,
    #                                           motif_len, Transition.D2M]

    # if force_matching:
    #     force_loss = np.log((match_cost+1)*match_cost/2) + \
    #         torch.sum((match_cost-1) * a[:, :, Transition.M2M], dim=1).mean()
    #     return - force_loss - torch.logsumexp(f[:, :, motif_len, random_len], dim=1).mean()
    # return - torch.logsumexp(f[:, :, motif_len, random_len], dim=1).mean()

class VAE(nn.Module):
    loss_fn: Callable[..., Tensor]
    # 再構成誤差項と正則化項を足し合わせて返却するか，別々に返却するか。
    embed_size: int
    # 埋め込み空間の次元数。

    def __init__(self, encoder, decoder, embed_size=10, hidden_size=32):
        super(VAE, self).__init__()
        self.embed_size = embed_size

        self.encoder = encoder
        self.decoder = decoder

        self.h2mu = nn.Linear(hidden_size, embed_size)
        self.h2logvar = nn.Linear(hidden_size, embed_size)

    def reparameterize(self, mu, logvar, deterministic=False):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + (std * eps if not deterministic else 0)
        return z

    def forward(self, input, deterministic=False):
        h = self.encoder(input)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)

        z = self.reparameterize(mu, logvar, deterministic)
        recon_param = self.decoder(z)
        return recon_param, mu, logvar

class EncoderCNN (nn.Module):
    # 0~3 is already used by embedding ATGC
    def __init__(self, embedding_dim=32, window_size=7, num_layers=6):
        super(EncoderCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        # self.embed = nn.Embedding(
        #     num_embeddings = 5,  # [A,T,G,C,PAD]
        #     embedding_dim = embedding_dim,
        #     padding_idx = NucleotideID.PAD,
        # )

        self.embed = nn.Embedding(
            num_embeddings=4,  # [A,T,G,C,PAD,SOS,EOS]
            embedding_dim=embedding_dim)

        modules = [Bottleneck(embedding_dim, window_size)
                   for _ in range(num_layers)]
        self.resnet = nn.Sequential(*modules)

    def forward(self, seqences):
        # change X from (N, L) to (N, L, C)
        x = F.leaky_relu(self.embed(seqences))

        # change X to (N, C, L)
        x = x.transpose(1, 2)
        value, indices = self.resnet(x).max(dim=2)
        return value

class DecoderPHMM(nn.Module):
    # tile hidden and input to make x
    def __init__(self, motif_len, embed_size, hidden_size=32):
        super(DecoderPHMM, self).__init__()

        class View(nn.Module):
            def __init__(self, shape):
                super(View, self).__init__()
                self.shape = shape

            def forward(self, x):
                return x.view(*self.shape)

        self.fc1 = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size*2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.tr_from_M = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, (motif_len+1)*3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len+1, 3)),
            nn.LogSoftmax(dim=2)
        )
        self.tr_from_I = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, (motif_len+1)*2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len+1, 2)),
            nn.LogSoftmax(dim=2)
        )
        self.tr_from_D = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, (motif_len+1)*2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len+1, 2)),
            nn.LogSoftmax(dim=2)
        )

        self.emission = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, motif_len*4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            View((-1, motif_len, 4)),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, input):
        x = self.fc1(input)

        transition_from_match = self.tr_from_M(x)
        transition_from_insertion = self.tr_from_I(x)
        transition_from_deletion = self.tr_from_D(x)

        emission_proba = self.emission(x)
        return (torch.cat((
            transition_from_match,
            transition_from_insertion,
            transition_from_deletion), dim=2), emission_proba)

class Bottleneck(nn.Module):
    def __init__(self, init_dim=32, window_size=7):
        super(Bottleneck, self).__init__()
        assert window_size % 2 == 1, f"window size should be odd, given {window_size}"

        self.conv1 = nn.Conv1d(
            in_channels=init_dim,
            out_channels=init_dim*2,
            kernel_size=1)

        self.conv2 = nn.Conv1d(
            in_channels=init_dim*2,
            out_channels=init_dim*2,
            kernel_size=window_size,
            padding=window_size//2
        )

        self.conv3 = nn.Conv1d(
            in_channels=init_dim*2,
            out_channels=init_dim,
            kernel_size=1)

        self.bn1 = nn.BatchNorm1d(init_dim)
        self.bn2 = nn.BatchNorm1d(init_dim*2)
        self.bn3 = nn.BatchNorm1d(init_dim*2)

    def forward(self, input):
        x = self.conv1(F.leaky_relu(self.bn1(input)))
        x = self.conv2(F.leaky_relu(self.bn2(x)))
        x = self.conv3(F.leaky_relu(self.bn3(x)))
        return F.leaky_relu(x+input)

class CNN_PHMM_VAE(VAE):
    def __init__(self, motif_len=12, embed_size=10, hidden_size=32, kernel_size=7):
        encoder = EncoderCNN(hidden_size, kernel_size)
        decoder = DecoderPHMM(motif_len, embed_size)

        super(CNN_PHMM_VAE, self).__init__(
            encoder, decoder, embed_size, hidden_size)
        self.loss_fn = profile_hmm_vae_loss

def embed_sequences(
    sequences: List[str], 
    model: VAE
    ) -> List[np.ndarray]:
    """対応する埋め込み空間の値を返却する。
    
    Parameters
    ----------
    sequences : List[str]
        各残基が `A, U, C, G` で表現された塩基配列のリスト。
    model : VAE
        埋め込みに使用する VAE モデル。
    
    Returns
    -------
    coords : List[np.ndarray]
        `sequences` に対応する埋め込み値
    """
    assert(type(sequences) == list)
    
    # https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
    model_device = next(model.parameters()).device
    with torch.no_grad():
        model.eval()
        coords: List[np.ndarray] = list()
        for sequence in sequences:
            recon, mu, logvar = model(
                torch.Tensor(
                    [ID_encode(sequence)],
                    device=model_device).long()
                )
            mu_np = mu.to('cpu').detach().numpy().copy()
            coords.append(mu_np)
    
    return coords

def get_most_probable_seq(
    coords: List[np.ndarray],
    model: VAE,
    ) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
    """座標に対応する pHMM モデルにおいて，最も生成されやすい配列を計算する。
    配列生成時の状態パスも書き出す。
    
    Parameters
    ----------
    coords : List[np.ndarray]
        座標値のリスト。
    model : VAE
        埋め込み値から pHMM のパラメータを計算する VAE モデル。
    
    Returns
    -------
    seqs_states_tuple : Tuple[List[str], List[List[Tuple[int, int]]]] 
        一つ目は最も生成されやすい配列を順に計算したリスト，二つ目は一つ目のリストが生成されるときの最適状態列。タプルの一つ目が配列の座位で，二つ目が `State`。
    """
    assert(np.array(coords).shape[1] == model.embed_size)

    sequences = list()
    state_transits = list()

    for coord in coords:
        transition_prob, emission_prob \
            = model.decoder(torch.Tensor([coord]))
        a_log_prob: np.ndarray = transition_prob.detach().numpy()[0]
        e_log_prob: np.ndarray = emission_prob.detach().numpy()[0]
        a_prob: np.ndarray = np.log(a_log_prob)
        e_prob: np.ndarray = np.log(e_log_prob)
        e_prob = e_prob / np.sum(e_prob, axis=1)[:, None]

        idx, state = 0, State.M
        states: List[Tuple[int, int]] = [(idx, state)]
        seq = ""
        while True:
            if state == State.M:
                p = a_prob[idx][np.array([
                    Transition.M2M.value,
                    Transition.M2I.value,
                    Transition.M2D.value])]
            elif state == State.I:
                p = [
                    a_prob[idx][Transition.I2M.value],
                    0,
                    0]
            elif state == State.D:
                p = [
                    a_prob[idx][Transition.D2M.value],
                    0,
                    a_prob[idx][Transition.D2D.value]]
            else:
                raise Exception()

            p[np.argmax(p)] += 1000000
            state = np.random.choice([State.M, State.I, State.D], p=p/sum(p))
            if state != State.I:
                idx += 1
            states.append((idx, state))

            if idx == a_prob.shape[0]:
                break

            if state == State.M:
                # logger.info("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(*self.e[idx-1]))
                p = np.copy(e_prob[idx-1])
                p[np.argmax(p)] += 100000
                seq += np.random.choice(list("ATGC"), p=p/sum(p))
            elif state == State.I:
                seq += "N"
            else:
                seq += "_"
        sequences.append(seq)
        state_transits.append(states)

    return sequences, state_transits
