# FCMCC
Current unsupervised time series clustering methods often struggle to fully exploit the inherent characteristics of time series data and commonly adopt a two-stage training strategy that separates feature learning from the clustering process. To address these limitations, this paper proposes a novel deep clustering framework, Time-Frequency augmented Multi-level Contrastive Clustering (TFMCC). TFMCC employs a multi-scale time-frequency augmentation strategy, where each training iteration stochastically selects time and frequency scales to generate diverse augmented views, enhancing the model’s ability to learn robust and generalizable representations. In addition, a multi-level contrastive learning mechanism is introduced to jointly capture temporal dependencies, inter-sample similarities, and cluster structures. By jointly optimizing these components, TFMCC enables the learning of temporally-aware and clustering-friendly representations.

## Requirements

The recommended requirements for FCMCC are specified as follows:
Python 3.9
* torch==2.8.1
* scipy==1.6.1
* numpy==1.26
* pandas==2.23

## Dataset Preparation

Before running TFMCC, it is essential to prepare the dataset. The time series dataset used in this project is sourced from the [UCR archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).

The datasets can be obtained and put into `datasets/` folder in the following way:

* [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into `datasets/UCR/` so that each data file can be located by `datasets/UCR/<dataset_name>/<dataset_name>_*.csv`.


## Usage
The script **batch_run_single.py** allows you to run TFMCC across multiple UCR datasets in a single command.  
All key hyperparameters (learning rate, batch size, temperatures, levels, etc.) can be configured directly in the script.

Once executed, it will:

- Automatically iterate over the selected datasets  
- Train TFMCC with the specified settings  
- Save the results (ACC, NMI, ARI, RI, FMI) for each dataset  
- Log the execution time for every run

To start batch evaluation, simply run:

```bash
python batch_run_single.py
