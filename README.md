# workload_prediction
This repository contains code for predicting CPU utilization, Memory Utilization using an attention-based LSTM/Bi-LSTM/GRU model and ARIMA model. The experiment was conducted in Google Colab Pro environment.


It contains following folders:

-The 'code' folder contains all the code files along with the respective datasets and model architectures.

-The 'Result_Graph' folder contains all the graphs generated from the experiments that we had performed. Each graph corresponds to a specific experiment and provides insights into the model's performance.

-The 'preprocessing' folder contains the code for the preprocessing methods we have used on the available raw dataset.

-The 'demo' folder contains a very small subset of the pre-processed dataset and an executed demo in .ipynb file.



## Requirements

- Python 3.10.12
- Keras 2.15.0
- NumPy 1.25.2
- Pandas 2.0.3
- Matplotlib 3.7.1
- PyKalman 0.9.7


## Installation

You can install the required packages using the following commands:

```bash
!pip install keras
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install pykalman
```

## Usage
Clone this repository.

Install the required packages as mentioned above.

Run the provided Jupyter Notebook to execute the experiment.


## Datasets
We have used publicly available datasets, the links to which can be found below:
1.Alibaba(2018): https://github.com/alibaba/clusterdata/tree/master/
cluster-trace-v2018. Accessed: April 20, 2024 (2024)
2.Materna: http://gwa.ewi.tudelft.nl/datasets/gwa-t-13-materna.
3.PlanetsLab: https://github.com/beloglazov/planetlab-workload-traces.
4.MicrosoftAzure: https://github.com/Azure/AzurePublicDataset.
5.Bitbrains: http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains.


