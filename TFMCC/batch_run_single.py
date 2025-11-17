import os
import subprocess
import pandas as pd
import time


def load_existing_results(results_path):
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    else:
        return pd.DataFrame(columns=['Dataset', 'ACC', 'NMI', 'ARI', 'RI', 'FMI'])


def load_time_log(time_log_path):
    if os.path.exists(time_log_path):
        return pd.read_excel(time_log_path)
    else:
        return pd.DataFrame(
            columns=['Dataset', 'Batch Size', 'Learning Rate', 'Seed', 'Start Time', 'End Time', 'Duration (s)'])


# Define dataset, learning rate, and batch size parameters
data_dir = "./datasets/UCR"
# data_dir = "./datasets/UEA"

data_dir_list = os.listdir(data_dir)
data_dir_list.sort()
datasets = data_dir_list

learning_rates = [0.001] #0.001
batch_sizes = [256]
seeds = [666]
alphas = [1]
epochs = [1400]

temporal_temperatures = [0.2]
instance_temperatures = [0.2]
cluster_temperatures = [0.5]
run_times = [1,]
change_factors = [0.8]
max_levels = [12] #


union_list = [
    'CricketZ', 'ToeSegmentation2', 'GestureMidAirD3', 'FaceAll', 'ShapeletSim', 'Lightning7',
    'FordB', 'ECGFiveDays', 'ShakeGestureWiimoteZ', 'Symbols', 'SonyAIBORobotSurface2', 'ProximalPhalanxTW',
    'GunPointOldVersusYoung', 'MiddlePhalanxOutlineAgeGroup', 'Beef', 'SyntheticControl', 'WordSynonyms',
    'ShapesAll', 'FacesUCR', 'SmoothSubspace', 'SwedishLeaf', 'TwoPatterns', 'MelbournePedestrian',
    'CricketX', 'MoteStrain', 'GesturePebbleZ1', 'AllGestureWiimoteX', 'GesturePebbleZ2', 'OSULeaf', 'DodgerLoopDay',
    'CBF',
    'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'SonyAIBORobotSurface1',

    'Meat', 'CricketY', 'Car',

    'FaceFour', 'FiftyWords', 'PickupGestureWiimoteZ',
]


# Time log file
time_log_path = "execution_time_log2.xlsx"
time_log_df = load_time_log(time_log_path)

# Iterate through all combinations
for run_time in run_times:
    for a1 in [-1]:
        for change_factor in change_factors:
            for batch_size in batch_sizes:
                for seed in seeds:
                    for lr in learning_rates:
                        for alpha in alphas:
                            for temporal_temperature in temporal_temperatures:
                                for instance_temperature in instance_temperatures:
                                    for cluster_temperature in cluster_temperatures:
                                        for max_level in max_levels:
                                            for epoch in epochs:
                                                for dataset in datasets:

                                                    if dataset not in union_list:
                                                        print(f"Skipping dataset: {dataset}")
                                                        continue

                                                    run_name = f"{dataset}_lr{lr}_bs{batch_size}"
                                                    output_name = f"test_{max_level}_{batch_size}_{lr}_{seed}_{alpha}_{epoch}_{temporal_temperature}_{instance_temperature}_{cluster_temperature}_dtw随机_{change_factor}_40_{run_time}"#{run_time}#pool512_簇级对比头_x归一化
                                                    output_dir = f"{output_name}.csv"

                                                    if not os.path.exists(f"./log/{output_name}"):
                                                        os.makedirs(f"./log/{output_name}")
                                                    output_csv = f"./log/{output_name}/{dataset}.csv"

                                                    existing_results_df = load_existing_results(output_dir)
                                                    completed_datasets = set(existing_results_df['Dataset'])

                                                    if dataset in completed_datasets:
                                                        print(f"Skipping already completed dataset: {dataset}")
                                                        continue

                                                    print(f"Running with dataset={dataset}, lr={lr}, batch_size={batch_size}")

                                                    # Record start time
                                                    start_time = time.time()
                                                    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))

                                                    # Construct command
                                                    cmd = [
                                                        "python", "train.py",
                                                        "--dataset", dataset,
                                                        "--run_name", run_name,
                                                        "--loader", "UCR", #"UEA"  "UCR"
                                                        "--gpu", "0",
                                                        "--batch-size", str(batch_size),
                                                        "--lr", str(lr),
                                                        '--epochs', str(epoch),
                                                        "--seed", str(seed),
                                                        "--eval",
                                                        "--output_dir", str(output_dir),
                                                        "--alpha", str(alpha),
                                                        "--max_level", str(max_level),
                                                        "--change_factor", str(change_factor),
                                                        "--temporal_temperature", str(temporal_temperature),
                                                        "--instance_temperature", str(instance_temperature),
                                                        "--cluster_temperature", str(cluster_temperature),
                                                        "--output_csv", str(output_csv)
                                                    ]

                                                    try:
                                                        subprocess.run(cmd, check=True)
                                                    except subprocess.CalledProcessError as e:
                                                        print(f"Error while running {run_name}: {e}")

                                                    # Record end time
                                                    end_time = time.time()
                                                    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
                                                    duration = round(end_time - start_time, 2)

                                                    # Log execution time
                                                    new_entry = pd.DataFrame([{
                                                        "Dataset": dataset,
                                                        "Batch Size": batch_size,
                                                        "Learning Rate": lr,
                                                        "Seed": seed,
                                                        "Start Time": start_time_str,
                                                        "End Time": end_time_str,
                                                        "Duration (s)": duration
                                                    }])

                                                    time_log_df = pd.concat([time_log_df, new_entry], ignore_index=True)
                                                    time_log_df.to_excel(time_log_path, index=False)

print("Execution time logging complete.")
