"""
Executing Linear Regression
------
I have created this file to execute linear regression.

It is similar to the other run scripts, this script allows to (1) train the model, (2) generate predictions, and (3) compute inventory decisions using
the queuing model described in Section 3.1 of the original paper on an arbitrary number of stations.
"""

from __future__ import print_function
import argparse
import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import pyro
import datetime
from workalendar.usa.new_york import NewYork
from pyro.infer import SVI, Trace_ELBO, Predictive
from src.algos.vprnn import VPRNN
import src.algos.inventory_decision_hourly as idh
import src.algos.inventory_decision_quarterly as idq
from src.misc.utils import get_performance_metrics, Trace_ELBO_Wrapper, read_and_preprocess_data, get_results
from sklearn.linear_model import LinearRegression


parser = argparse.ArgumentParser(description='Full pipeline example')
# RNN parameters
parser.add_argument('--epochs', type=int, default=50000, metavar='N',
                    help='number of epochs to train (default: 50k)')
parser.add_argument('--no-cuda', type=bool, default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--patience', type=int, default=1000, metavar='N',
                    help='how many epochs without improvement to stop training')
parser.add_argument('--no-train', type=bool, default=False,
                    help='disables training process')
parser.add_argument('--no-predict', type=bool, default=False,
                    help='collects pre-computed prediction')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')

# Data parameters
parser.add_argument('--stations', default=[426], nargs='+',
                    help='list of station IDs on which to run pipeline')
parser.add_argument('--interval', type=int, default=60, metavar='S',
                    help='defines temporal aggregation (defaul 60min)')

# Queuing model parameters
parser.add_argument('--no-decision', default=False, action='store_true',
                    help='disables decision model')
parser.add_argument('--benchmark', action='store_true',
                    help='enables benchmark decision model')

# Parse and preprocess input arguments
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.decision = not args.no_decision
args.train = not args.no_train
args.predict = not args.no_predict
args.interval = args.interval
args.directory = args.directory + f"/{args.interval}min"
if args.interval in [15, 30]:
    args.file_interval = str(args.interval) + 'min'
if args.interval == 60:
    args.file_interval = 'hourly'

# Fix random seed for reproducibility     
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

if args.stations[0] == 'all':
    args.stations = [128, 151, 168, 229, 285, 293, 327, 358, 359, 368, 
                    387, 402, 426, 405, 435, 445, 446, 453, 462, 482, 
                    491, 497, 499, 504, 514, 519, 3263, 3435, 3641, 3711
                    ]

# define test dates
start_date = datetime.date(2018,11,1)
end_date = datetime.date(2018,12,31)
date_list = []
for n in range((end_date - start_date).days + 1):
    dt = start_date + datetime.timedelta(days=n)
    date_list.append(dt)

# loop over selected stations
for station in args.stations:
    print('now running ',station)
    try:
        # Load dataset
        df_station, X_train, X_valid, X_test, y_train, y_valid, y_test, X_tensor, y_tensor = \
                 read_and_preprocess_data(demand_path=f"data/demand_rate/{str(args.interval)}min/{str(station)}_{args.file_interval}RatesByDay_2018.csv",
                                        weather_path=f"data/raw/weather2018_{args.interval}min.csv", station_229=False, interval=args.interval)
        labels = ['return', 'pickup']
        # If train==True, start training loop through ELBO maximization (Section 3.2)
        print('checking if training is enabled',args.train)
        if args.train:
            print('attempting to train for station ',station)
            for i, label in enumerate(labels):
                print(f"\n Training started for St. {station} ({label}), with patience={args.patience}")
                # select return/pickup sequence
                y_train_i, y_valid_i, y_test_i, y_tensor_i = y_train[:,i][:,None], y_valid[:,i][:,None], y_test[:,i][:,None], y_tensor[:,i][:,None]
                X_train_i, X_valid_i, X_test_i, X_tensor_i = X_train, X_valid, X_test, X_tensor
                # train process
                
                lr = LinearRegression()

                

                train_losses = []
                valid_losses = []

        # If predict==True, generate predictions for test data
        if args.predict:
            for i, label in enumerate(labels):
                y_train_i, y_valid_i, y_test_i, y_tensor_i = y_train[:,i][:,None], y_valid[:,i][:,None], y_test[:,i][:,None], y_tensor[:,i][:,None]
                X_train_i, X_valid_i, X_test_i, X_tensor_i = X_train, X_valid, X_test, X_tensor
                # create model instance and load optimal pre-trained parameters
                lr.fit(X_train_i, y_train_i)
                print('generating results ',label)
                get_results(model=lr, X=X_tensor_i, y=y_tensor_i, station=station, results_path=args.directory, interval=args.interval,
                            model_type="lr", labels=[label], write_mode="w" if i==0 else "a")
                print('completed result generation')


############################################################
################ decision pipeline from here ###############
############################################################

        # if decision==True, compute inventory decisions through queuing model (Section 3.1)
        if args.decision:
            # if interval==60min, use hourly predictions as inputs
            if args.interval == 60:
                idh.get_rnn_inventory_decisions(station_id=station, date_list=date_list, hour_range=range(0,24), model_type=['so_rnn',], data_dir='data', prediction_dir=f'{args.directory}/predicted_demand/', result_dir=args.directory)
                if args.benchmark:
                    idh.get_rnn_inventory_decisions(station_id=station, date_list=date_list, hour_range=range(0,24), model_type=['poisson_rnn', 'lr'], data_dir='data', prediction_dir=f'{args.directory}/predicted_demand/', result_dir=args.directory)
                    idh.get_benchmark_inventory_decisions(station, date_list, hour_range=range(0,24), data_dir='data', result_dir=args.directory)
                print(f'Station {station} decision calculation finished!')
                # Evaluation  
                idh.get_inventory_decision_evaluation_results(station, date_list, hour_range=range(0,24), model_type=['so_rnn',], flag_benchmark=args.benchmark, data_dir='data', result_dir=args.directory)
            else:
            # interval==15min or 30min
                idq.get_rnn_inventory_decisions(station_id=station, date_list=date_list, hour_range=range(0,24), quarter=args.interval, model_type=['so_rnn',], data_dir='data', prediction_dir=f'{args.directory}/predicted_demand/', result_dir=args.directory)
                if args.benchmark:
                    idq.get_rnn_inventory_decisions(station_id=station, date_list=date_list, hour_range=range(0,24), quarter=args.interval, model_type=['poisson_rnn', 'lr'], data_dir='data', prediction_dir=f'{args.directory}/predicted_demand/', result_dir=args.directory)
                    idq.get_benchmark_inventory_decisions(station, date_list, hour_range=range(0,24), quarter=args.interval, data_dir='data', result_dir=args.directory)
                print(f'Station {station} decision calculation finished!')
                # Evaluation  
                idq.get_inventory_decision_evaluation_results(station, date_list, hour_range=range(0,24), quarter=args.interval, model_type=['so_rnn',], flag_benchmark=args.benchmark, data_dir='data', result_dir=args.directory)
    except Exception as e:
        print(e)