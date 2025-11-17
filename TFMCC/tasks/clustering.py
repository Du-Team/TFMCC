import os

import numpy as np
import pandas as pd
from skfuzzy import cmeans_predict
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, accuracy_score, fowlkes_mallows_score
)
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment  # 使用 scipy 替代 sklearn
from skfuzzy.cluster import cmeans
from scipy.special import comb

# ACC 需要通过匈牙利算法重新对齐标签
def calculate_acc(true_labels, predicted_labels):
    # 获取唯一标签
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)

    cost_matrix = np.zeros((len(unique_true), len(unique_pred)))
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            cost_matrix[i, j] = np.sum((true_labels == true_label) & (predicted_labels == pred_label))

    # 使用匈牙利算法进行最佳匹配
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)  # 改为使用 scipy 的 linear_sum_assignment
    matched_labels = predicted_labels.copy()
    for i, j in zip(row_ind, col_ind):
        matched_labels[predicted_labels == unique_pred[j]] = unique_true[i]

    return accuracy_score(true_labels, matched_labels)

## juge measure
def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def eval_clustering_old(dataset, model, train_data, train_labels, test_data, test_labels, n_clusters=None):
    assert train_labels.ndim == 1 or train_labels.ndim == 2

    # 编码表示
    """
    output:
        encoding_window='full_series':
            表示对整个时间序列进行池化，因此返回的形状是 (batch_size, output_features) #n_timestamps维度被最大池化掉了
        encoding_window='multiscale':
            表示多尺度池化，返回的形状会是 (n_instance, n_timestamps, pooled_features)，这里的 pooled_features 取决于具体多尺度池化产生的特征数
        encoding_window = 一个整数:
            表示基于指定的窗口大小进行池化，返回的形状是 (n_instance, n_timestamps, n_features)
    """
    # train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    # 设置聚类的类簇数量，如果未指定则使用标签的类别数
    if n_clusters is None:
        n_clusters = len(np.unique(train_labels))


    method = "kmeans"

    # 判断使用 KMeans 或 FCM 聚类
    if method == 'kmeans':
        # 训练 KMeans 聚类模型
        # kmeans = KMeans(n_clusters=n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(test_repr)

        # 预测标签
        # train_pred = kmeans.predict(train_repr)
        test_pred = kmeans.predict(test_repr)
    elif method == 'fcm':
        # 使用 KMeans++ 初始化聚类中心
        print("Initializing FCM with K-Means++ cluster centers...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(test_repr)  # 在训练集上运行 KMeans
        initial_centers = kmeans.cluster_centers_  # shape: (n_clusters, features)

        # 计算距离矩阵 (samples x clusters)
        distances = np.linalg.norm(test_repr[:, np.newaxis] - initial_centers, axis=2)  # shape: (samples, clusters)

        # 计算初始隶属度矩阵 u_initial
        inv_distances = 1 / (distances + 1e-10)  # 避免除零
        inv_distances_m = inv_distances ** (2 / (1.5 - 1))  # 使用模糊指数调整，m=1.5
        u_initial = inv_distances_m / np.sum(inv_distances_m, axis=1, keepdims=True)  # 归一化
        u_initial = u_initial.T  # 转置为 (clusters, samples)


        # 应用 Fuzzy C-Means (FCM) 聚类
        print("Applying Fuzzy C-Means (FCM) clustering on training data...")
        cntr, u, u0, d, jm, p, fpc = cmeans(
            test_repr.T,  # 数据矩阵，形状为 (features, samples)
            c=n_clusters,  # 聚类数量
            m=1.5,         # 模糊系数
            error=0.005,   # 收敛误差阈值
            maxiter=1000,  # 最大迭代次数
            init=u_initial  # 使用初始化的隶属度矩阵
        )

        # cntr, u, u0, d, jm, p, fpc = cmeans(train_repr.T, n_clusters, m=1.5, error=0.005, maxiter=1000)
        # train_pred = np.argmax(u, axis=0)

        # 使用得到的中心预测测试数据
        u_test, u0_test, d_test, jm_test, p_test, fpc_test = cmeans_predict(test_repr.T, cntr, m=1.5, error=0.005,
                                                                            maxiter=1000)
        test_pred = np.argmax(u_test, axis=0)

    else:
        raise ValueError("Unsupported clustering method. Use 'kmeans' or 'fcm'.")

    logdir = f"result"
    # 保存为csv文件
    save_dir = f"./{logdir}/label/"
    os.makedirs(save_dir, exist_ok=True)  # 如果路径不存在则创建

    # 将label_true转换为DataFrame并保存为CSV
    df_true = pd.DataFrame(test_labels, columns=['label_true'])
    df_true.to_csv(f'./{logdir}/label/{dataset}_label_true.csv', index=False)

    # 将label_pred转换为DataFrame并保存为CSV
    df_pred = pd.DataFrame(test_pred, columns=['label_pred'])
    df_pred.to_csv(f'./{logdir}/label/{dataset}_label_pred.csv', index=False)

    # 计算评估指标：
    # 1. 聚类准确度 (ACC)
    # train_acc = calculate_acc(train_labels, train_pred)
    test_acc = calculate_acc(test_labels, test_pred)

    # 2. 调整兰德指数 (ARI)
    # train_ari = adjusted_rand_score(train_labels, train_pred)
    test_ari = adjusted_rand_score(test_labels, test_pred)

    # 3. 归一化互信息 (NMI)
    # train_nmi = normalized_mutual_info_score(train_labels, train_pred)
    test_nmi = normalized_mutual_info_score(test_labels, test_pred)

    # 4. Rand Index (RI) 是 Rand 指数的标准版本，通常与 ARI 相比
    # 在 sklearn 中没有直接的 RI，但 RI 可以通过调整 ARI 公式计算。
    # RI = (TP + TN) / (TP + TN + FP + FN)
    # 简化为 RI = (ARI + 1) / 2 (近似公式)
    # train_ri = (train_ari + 1) / 2
    # test_ri = (test_ari + 1) / 2

    test_ri = rand_index_score(test_pred, test_labels)
    # 5. Fowlkes-Mallows Index (FMI)
    # train_fmi = fowlkes_mallows_score(train_labels, train_pred)
    test_fmi = fowlkes_mallows_score(test_labels, test_pred)

    # 返回聚类结果的评估指标
    return test_acc, test_nmi, test_ari, test_ri, test_fmi

def eval_clustering(dataset, model, test_data, test_labels, logdir=None, n_clusters=None, save_end = False):
    assert test_labels.ndim == 1 or test_labels.ndim == 2

    test_pred, all_embeddings, all_out_c = model.predict_clusters(test_data, save_end)



    if save_end == True:
        # logdir = f"result"
        # 保存为csv文件
        save_dir = f"./label/{logdir}"
        os.makedirs(save_dir, exist_ok=True)  # 如果路径不存在则创建

        # 将label_true转换为DataFrame并保存为CSV
        df_true = pd.DataFrame(test_labels, columns=['label_true'])
        df_true.to_csv(f'{save_dir}/{dataset}_label_true.csv', index=False)

        # 将label_pred转换为DataFrame并保存为CSV
        df_pred = pd.DataFrame(test_pred, columns=['label_pred'])
        df_pred.to_csv(f'{save_dir}/{dataset}_label_pred.csv', index=False)

        save_dir2 = f"./rep/{logdir}"
        os.makedirs(save_dir2, exist_ok=True)  # 如果路径不存在则创建
        # 如果需要保存表征，保存为 .npy 文件
        np.save(f'{save_dir2}/{dataset}_rep.npy', all_embeddings)
        # print(f"Embeddings saved to {embedding_filename}")
        np.save(f'{save_dir2}/{dataset}_rep_out_c.npy', all_out_c)

    # 计算评估指标：
    # 1. 聚类准确度 (ACC)
    # train_acc = calculate_acc(train_labels, train_pred)
    test_acc = calculate_acc(test_labels, test_pred)

    # 2. 调整兰德指数 (ARI)
    # train_ari = adjusted_rand_score(train_labels, train_pred)
    test_ari = adjusted_rand_score(test_labels, test_pred)

    # 3. 归一化互信息 (NMI)
    # train_nmi = normalized_mutual_info_score(train_labels, train_pred)
    test_nmi = normalized_mutual_info_score(test_labels, test_pred)

    # 4. Rand Index (RI) 是 Rand 指数的标准版本，通常与 ARI 相比
    # 在 sklearn 中没有直接的 RI，但 RI 可以通过调整 ARI 公式计算。
    # RI = (TP + TN) / (TP + TN + FP + FN)
    # 简化为 RI = (ARI + 1) / 2 (近似公式)
    # train_ri = (train_ari + 1) / 2
    # test_ri = (test_ari + 1) / 2

    test_ri = rand_index_score(test_pred, test_labels)
    # 5. Fowlkes-Mallows Index (FMI)
    # train_fmi = fowlkes_mallows_score(train_labels, train_pred)
    test_fmi = fowlkes_mallows_score(test_labels, test_pred)

    # 返回聚类结果的评估指标
    return test_acc, test_nmi, test_ari, test_ri, test_fmi
