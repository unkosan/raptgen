#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations
# from typing import Annotated
from typing import Callable, Tuple, Type
import random
import numpy as np
import pandas as pd
from path import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

# seed 値の固定は https://qiita.com/north_redwing/items/1e153139125d37829d2d を参照した。
# 英文では https://pytorch.org/docs/stable/notes/randomness.html が相当する。

_seed_exists: bool = False
"""random seed が存在するかどうかを示すフラグ。`True` 時に存在を保証する。 """

_seed: int = 42
"""実際のシード値。デフォルトでは `42`。"""

def set_seed(seed: int):
    """シード値を設定する。この関数本体では DataLoader 周り以外のシード値を設定しているが，この関数が呼ばれると次に DataLoader が生成されたときもシード値が設定される。
    `get_dataloader` よりも前に本関数を実行すること。
    
    Parameters
    ----------
    seed : int
        seed 値
    
    Returns
    -------
    None
    """
    _seed_exists = True
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # "backends" is not a known member of module Pylance (reportGeneralTypeIssues) 
    # が下二行において発生するが，これは意図しているものらしい。詳しくは https://githubhot.com/repo/microsoft/pyright/issues/2232 を参照。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  
    random.seed(worker_seed)

def get_dataloader(
    ndarray_data: np.ndarray, # 配列を入力させてもいいが，各配列がちゃんとパディングされて一定の配列長になっていることを保証するため np.ndarray を入力させる。
    test_size: float = 0.1,
    batch_size: int = 512,
    train_test_shuffle: bool = True,
    use_cuda: bool = False, # refactoring 前はデフォルトで Cuda を使用している。フールプルーフでデフォルトを CPU 側にする。
    num_workers: int = 1,
    pin_memory: bool = False, # refactoring 前は GPU を使用する際デフォルトで True だった。確認していないが遺伝研の CPU で libgomp error 吐くならおそらくここが問題の可能性があるので今回はデフォルトで False にしている。ちなみにここを True にすると https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587#12-pin_memory の説明にあるとおり推論が早くなる。
    **kwargs
    ) -> Tuple[DataLoader, DataLoader]:
    """`ndarray_data` で提供される配列データをトレーニング用およびテスト用のデータセットに分割し，それぞれを `DataLoader` 形式で返却する。
    `set_seed(seed)` を実行した場合，分割結果も DataLoader のシードも `seed` で固定される。
    
    Parameters
    ----------
    ndarray_data : np.ndarray
        トレーニング用，テスト用に分割したいデータセット。
        各配列は `core.preprocessing.ID_encode()` で ID 化もしくは `core.preprocessing.one_hot_encode()` で one-hot 化されていなければならず，さらに全配列が同一配列長になるようにパディングされていなければならない。パディングを取り除く場合は `nn.Modules` によるモデル設計で行う。
    test_size : float, default = 0.1
        全体を 1 としたときのテスト用データの割合。
    batch_size : int, default = 512
        学習時のバッチの大きさ。
    train_test_shuffle : bool, default = True
        トレーニングデータ，テストデータを分割する際に元の `ndarray_data` に記述する順序を保持するかシャッフルするかを指定する。`True` 時は `ndarray_data` をシャッフルする。
    use_cuda : bool, default = False
        生成された DataLoader を使って CUDA を搭載した GPU 上で推論を行うかどうかを指定する。`True` で使用すると指定した際 `num_workers` 引数および `pin_memory` 引数が有効化される。
    num_workers : int, default = 1
        data loading を行う worker の数を指定する。`use_cuda = True` で CUDA が搭載された GPU 上で推論を行うと指定した際に有効になり，そうでない場合は `DataLoader` のコンストラクタで指定される `num_workers` は `0` となる。
    pin_memory : bool, default = False
        `True` の場合推論時に automatic memory pinning が適用され，計算が若干高速化される。詳しくは [ここ](https://qiita.com/sugulu_Ogawa_ISID/items/62f5f7adee083d96a587#12-pin_memory) を参照。
        `use_cuda = True` で CUDA が搭載された GPU 上で推論を行うと指定した際に有効になり，そうでない場合は `DataLoader` のコンストラクタで指定される `pin_memory` は `False` となる。
    
    Returns
    -------
    loaders : Tuple[DataLoader, DataLoader]
        タプルの中身は `train_loader, test_loader` であり，それぞれトレーニングデータ，テストデータの DataLoader である。
    """
    train_data, test_data = train_test_split(
        ndarray_data, test_size=test_size, shuffle=train_test_shuffle)
    train_data = TensorDataset(torch.from_numpy(train_data).long())
    test_data = TensorDataset(torch.from_numpy(test_data).long())

    if use_cuda == True:
        kwargs['num_workers'] = num_workers
        kwargs['pin_memory'] = pin_memory
    else:
        pass

    if _seed_exists == True:
        gen = torch.Generator()
        gen.manual_seed(_seed)
        kwargs['worker_init_fn'] = _seed_worker
        kwargs['generator'] = gen

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size, 
        shuffle=True,
        **kwargs
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        **kwargs)

    return train_loader, test_loader

def train(
    epochs: int,
    model: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_save_path: Path or str,
    loss_values_save_path: Path or str or None = None,
    device: torch.device = torch.device("cpu"), # フォールトトレランス。
    loss_fn: Callable or None = None,
    beta: int = 1,
    beta_threshold: int = 20,
    beta_schedule: bool = False,
    force_matching: bool = False,
    force_epochs: int = 20,
    logs: bool = True,
    position: int = 0,
    ) -> pd.DataFrame:
