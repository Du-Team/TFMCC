import pandas as pd
import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime

from tasks.clustering import eval_clustering, eval_clustering_old  # ,eval_clustering_old
from tfmcc import tfmcc
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

def save_checkpoint_callback( #在训练过程中定期保存模型状态
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

def load_existing_results(results_path):
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    else:
        return pd.DataFrame(columns=['Dataset', 'ACC', 'NMI', 'ARI', 'RI', 'FMI'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="TwoPatterns", help='The dataset name')
    parser.add_argument('--run_name', type=str, default="TwoPatterns", help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, default="UCR", help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=256, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=1400, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--output_dir', type=str, default=None, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--alpha', type=float, default=1.0,help='The ratio of alpha')
    parser.add_argument('--change_factor', type=float, default=0.8, help='The ratio of change_factor')
    parser.add_argument('--max_level', type=int, default=12, help='The max_level')
    parser.add_argument('--temporal_temperature', type=float, default=0.5, help='The ratio of temporal_temperature')
    parser.add_argument('--instance_temperature', type=float, default=0.5, help='The ratio of instance_temperature')
    parser.add_argument('--cluster_temperature', type=float, default=0.2, help='The ratio of cluster_temperature')
    parser.add_argument('--output_csv', type=str, default=None, help='The ratio of missing observations (defaults to 0)')
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'clustering'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        n_clusters = len(np.unique(train_labels))
    elif args.loader == 'UEA':
        task_type = 'clustering'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        n_clusters = len(np.unique(train_labels))
    else:
        raise ValueError(f"Unknown loader {args.loader}.")


        
    if args.irregular > 0:
        if task_type == 'classification':
            train_data = data_dropout(train_data, args.irregular)
            test_data = data_dropout(test_data, args.irregular)
        else:
            raise ValueError(f"Task type {task_type} is not supported when irregular>0.")
    print('done')

    combined_data = np.concatenate((train_data, test_data), axis=0)
    combined_labels = np.concatenate((train_labels, test_labels), axis=0)

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        alpha = args.alpha,
        temporal_temperature =  args.temporal_temperature,
        instance_temperature =  args.instance_temperature,
        cluster_temperature =  args.cluster_temperature,
        change_factor = args.change_factor,
        max_level = args.max_level,
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    t = time.time()
    
    model = tfmcc(
        input_dims=combined_data.shape[-1],
        class_num=n_clusters,
        device=device,
        **config
    )
    loss_log, results = model.fit(
        combined_data,
        combined_labels,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True,
        output_csv = args.output_csv,

    )
    # model.save(f'{run_dir}/model.pkl')
    # 保存评估指标到CSV文件
    if args.output_csv:
        results_df = pd.DataFrame(results, columns=["Epoch", "ACC", "NMI", "ARI", "RI", "FMI", "Loss"])
        results_df.to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        if task_type == 'clustering':
            print(f"n_clusters:{n_clusters}")

            # test_acc, test_nmi, test_ari, test_ri, test_fmi = eval_clustering_old(args.dataset, model, combined_data, combined_labels, combined_data, combined_labels, n_clusters)
            # print(f"old_compute:test_acc:{test_acc}, test_nmi:{test_nmi}, test_ari:{test_ari}, test_ri:{test_ri}, test_fmi:{test_fmi}")
            logdir = args.output_dir.rsplit(".", 1)[0]
            test_acc, test_nmi, test_ari, test_ri, test_fmi = eval_clustering(args.dataset, model, combined_data, combined_labels, logdir,  n_clusters,  save_end = True)
            # test_acc, test_nmi, test_ari, test_ri, test_fmi = eval_clustering_old(args.dataset, model, combined_data, combined_labels, combined_data, combined_labels, n_clusters)


            # print(clustering_results)
            print(f"new_test_acc:{test_acc}, test_nmi:{test_nmi}, test_ari:{test_ari}, test_ri:{test_ri}, test_fmi:{test_fmi}")
        else:
            assert False
        # pkl_save(f'{run_dir}/out.pkl', out)

        results_save = []

        output_dir = args.output_dir
        existing_results_df = load_existing_results(output_dir)

        results_save = existing_results_df.values.tolist()
        #dataname, acc, nmi,ari,ri,fmi
        results_save.append((args.dataset, test_acc, test_nmi, test_ari, test_ri, test_fmi))
        # 立即将更新的结果列表写入 CSV 文件
        results_df = pd.DataFrame(results_save, columns=['Dataset', 'ACC', 'NMI', 'ARI', 'RI', 'FMI'])
        results_df.to_csv(output_dir, index=False)

    print("Finished.")
