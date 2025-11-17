import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss, ClusterLoss
# from tools import adjust_learning_rate
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math

from tasks.clustering import eval_clustering

import pytorch_wavelets as pw
import random


def wavelet_scaling(data, change_factor=0.05, wavelet='db4', max_level=6):
    # 将输入数据转换为 PyTorch Tensor，并移动到 GPU（如果可用）
    nan_mask = ~data.isnan().any(axis=-1)
    data[~nan_mask] = 0

    # level = 5
    level = random.randint(2, max_level)  # [a, b] 1, max_level
    # print(f'level:{level}')
    scalinglevel = random.randint(0, level - 1)

    scale_factor = random.uniform(1 - change_factor, 1 + change_factor)
    # print(f"scalinglevel: {scalinglevel}, scale_factor: {scale_factor}")

    # 创建1D小波变换前向分解对象
    dwt_forward = pw.DWT1DForward(J=level, wave=wavelet, mode='zero').to(data.device)

    # 创建1D小波逆变换对象
    dwt_inverse = pw.DWT1DInverse(wave=wavelet, mode='zero').to(data.device)

    # 执行前向小波变换
    low_coeffs, high_coeffs = dwt_forward(data.transpose(1, 2))  # 数据形状: (N, C, L) -> (1, 1, L)

    # 对细节系数进行缩放
    high_coeffs[scalinglevel] = high_coeffs[scalinglevel] * scale_factor

    # 执行逆小波变换重构信号
    reconstructed = dwt_inverse((low_coeffs, high_coeffs))
    reconstructed = reconstructed[:, :, :data.shape[1]].transpose(1, 2)

    reconstructed[~nan_mask] = 0
    # 确保输出长度与输入一致
    return reconstructed


class tfmcc:
    def __init__(
            self,
            input_dims,
            class_num,
            output_dims=320,
            hidden_dims=64,
            depth=10,
            device='cuda',
            lr=0.001,
            batch_size=16,
            alpha=1.0,
            change_factor=0.8,
            temporal_temperature=1.0,
            instance_temperature=1.0,
            cluster_temperature=1.0,
            max_train_length=None,
            temporal_unit=0,
            after_iter_callback=None,
            after_epoch_callback=None,
            max_level=6,
            len=0.5,
    ):


        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.class_num = class_num
        self.alpha = alpha

        self._net = TSEncoder(input_dims=input_dims, class_num=class_num, output_dims=output_dims,
                              hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0  # n_iters：表示训练过程中总的迭代次数（每次迭代处理一个 batch），包含所有的 batch（跨越所有的 epoch）。它用于控制总的训练迭代次数。
        # 代码运行的总迭代次数n_iters是固定的，batchsize越大每次迭代的计算量就越大

        self.temporal_temperature = temporal_temperature  # 1.0
        self.instance_temperature = instance_temperature  # 1.0
        self.cluster_temperature = cluster_temperature  # 1.0 #0.5

        self.change_factor = change_factor
        self.max_level = max_level
        self.len = len

    def fit(self, train_data, train_labels, n_epochs=None, n_iters=None, verbose=False, output_csv=None):
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            # n_iters = 200 if train_data.size <= 100000 else 600
            n_iters = 800 if train_data.size <= 100000 else 2400

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        # 可用
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True,
                                  drop_last=True)

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        criterion_cluster = ClusterLoss(self.class_num, self.cluster_temperature, self.device).to(self.device)

        loss_log = []
        results = []  # 用于保存每个epoch的结果

        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset: window_offset + self.max_train_length]
                x = x.to(self.device)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
                # print(f"len:{self.len}")
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

                optimizer.zero_grad()

                x_left = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)

                x_right_aug = wavelet_scaling(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left),
                                              change_factor=self.change_factor, max_level=self.max_level)



                out1, out1_c = self._net(x_left)
                out1 = out1[:, -crop_l:]

                out2, out2_c = self._net(x_right_aug)
                out2 = out2[:, :crop_l]

                loss_r = hierarchical_contrastive_loss(out1, out2, self.instance_temperature, self.temporal_temperature,
                                                       temporal_unit=self.temporal_unit)
                loss_c = criterion_cluster(out1_c, out2_c)

                loss = loss_r + self.alpha * loss_c

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                n_epoch_iters += 1

                self.n_iters += 1

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            if interrupted:
                break

            # cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)



            self.n_epochs += 1

            test_acc, test_nmi, test_ari, test_ri, test_fmi = eval_clustering(None, self, train_data, train_labels,
                                                                              self.class_num, save_end=False)
            # 返回聚类结果的评估指标
            if verbose:
                # current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch #{self.n_epochs}: loss={cum_loss} acc:{test_acc}, nmi:{test_nmi}, ari:{test_ari}, ri:{test_ri}, fmi:{test_fmi}")

            if output_csv:
                results.append([self.n_epochs, test_acc, test_nmi, test_ari, test_ri, test_fmi, cum_loss])

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        return loss_log, results

    #

    # x: 输入张量，形状为 (batch_size, n_timestamps, n_features)。
    def _eval(self, x, mask=None, slicing=None, encoding_window=None):
        # output：(batch_size, n_timestamps, output_features)  default：output_dims=320
        out, out_c = self.net(x.to(self.device, non_blocking=True), mask)
        c = torch.argmax(out_c, dim=1)
        return c.cpu(), out.cpu(), out_c.cpu()

    def predict_clusters(self, data, save_end, batch_size=None, mask=None):
        '''
        Predict the clusters for the given data using the trained model.

        Args:
            data (numpy.ndarray): Input data to be predicted. Should have shape (n_instance, n_timestamps, n_features).
            batch_size (int, optional): Batch size to be used for inference. Defaults to training batch size if None.
            mask (str, optional): The mask used by the encoder. Defaults to None.

        Returns:
            np.ndarray: Predicted cluster indices for each instance.
        '''
        assert self.net is not None, 'Please train or load a net first'
        assert data.ndim == 3, "Input data should be of shape (n_instance, n_timestamps, n_features)"

        # 只有最后一轮才重新启动数据处理
        if save_end == True:
            if self.max_train_length is not None:
                sections = data.shape[1] // self.max_train_length
                if sections >= 2:
                    data = np.concatenate(split_with_nan(data, sections, axis=1), axis=0)

            temporal_missing = np.isnan(data).all(axis=-1).any(axis=0)
            if temporal_missing[0] or temporal_missing[-1]:
                data = centerize_vary_length_series(data)

            data = data[~np.isnan(data).all(axis=2).all(axis=1)]
            ######


        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        # Prepare the dataset and DataLoader for batch processing
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Put the model in evaluation mode
        org_training = self.net.training
        self.net.eval()

        predictions = []
        all_embeddings = []  # 用来保存所有的表征（嵌入）
        all_out_c = []

        with torch.no_grad():
            for batch in loader:
                x = batch[0]  # Get the input tensor from the batch
                cluster_preds, out, out_c = self._eval(x, mask=mask)  # Get the cluster predictions
                predictions.append(cluster_preds)
                all_embeddings.append(out.numpy())
                all_out_c.append(out_c.numpy())

        # Convert the list of predictions to a numpy array
        predictions = np.concatenate(predictions, axis=0)

        # 将表征（嵌入）转换为 numpy 数组
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_out_c = np.concatenate(all_out_c, axis=0)


        # Restore the model to training mode
        # 将模型恢复到推理之前的训练模式。如果 org_training 是 True，模型就会恢复到训练模式；如果是 False，则模型会恢复到评估模式。
        self.net.train(org_training)

        return predictions, all_embeddings, all_out_c

    def save(self, fn):
        ''' Save the model to a file.

        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)

    def load(self, fn):
        ''' Load the model from a file.

        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)

