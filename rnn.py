import pandas as pd
import string
import numpy as np
import torch.nn as nn
import torch
import sys
from sklearn.model_selection import train_test_split

from torch.autograd import Variable
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

HOSPITAL_DATASET_FILENAME = "combined-dataset.csv"

def data_preprocessing():
	hospital_data = pd.read_csv(HOSPITAL_DATASET_FILENAME)
	# removing dates
	hospital_data.drop(["date", "as_of_date"], axis='columns', inplace=True)

	X = hospital_data.drop("COVID_HOSP", axis="columns", inplace=False)
	y = hospital_data.loc[:, "COVID_HOSP"]

	X = pd.get_dummies(X, columns=["prname"])
	""" 
    # removing bad features
    X.drop(["reporting_week",
            "numcases_total",
            "numcases_weekly",
            "ratecases_total",
            "numdeaths_last14",
            "ratedeaths_last14",
            "avgincidence_last7",
            "avgdeaths_last7",
            "avgratedeaths_last7",
            "numtotal_all_distributed",
            "prname_British Columbia",
            "prname_Canada",
            "prname_Ontario",
            "prname_Quebec"], axis='columns', inplace=True)
    """
	return (X, y)

class Covid_LSTM(torch.nn.Module) :
	def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
		super().__init__()
		self.n_hidden = n_hidden
		self.seq_len = seq_len
		self.n_layers = n_layers
		self.lstm = nn.LSTM(
			input_size=n_features,
			hidden_size=n_hidden,
			num_layers=n_layers,
			dropout=0.5
		)
		self.linear = nn.Linear(in_features=n_hidden, out_features=1)

	def reset_hidden_state(self):
		self.hidden = (
			torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
			torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
		)
	def forward(self, sequences):
		# print(sequences.shape())
		lstm_out, self.hidden = self.lstm(
			sequences.view(len(sequences), self.seq_len, -1),
			self.hidden
		)
		last_time_step = \
			lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
		y_pred = self.linear(last_time_step)
		return y_pred

def train_model(
		model,
		train_data,
		train_labels,
		test_data=None,
		test_labels=None,
		lr=0.01
):
	loss_fn = torch.nn.MSELoss(reduction='sum')
	optimiser = torch.optim.Adam(model.parameters(), lr=lr)
	num_epochs = 60
	train_hist = np.zeros(num_epochs)
	test_hist = np.zeros(num_epochs)
	for t in range(num_epochs):
		model.reset_hidden_state()
		y_pred = model(train_data)
		loss = loss_fn(y_pred.float(), train_labels)
		if test_data is not None:
			with torch.no_grad():
				y_test_pred = model(test_data)
				test_loss = loss_fn(y_test_pred.float(), test_labels)
			test_hist[t] = test_loss.item()
			if t % 10 == 0:
				print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
		elif t % 10 == 0:
			print(f'Epoch {t} train loss: {loss.item()}')
		train_hist[t] = loss.item()
		optimiser.zero_grad()
		loss.backward()
		optimiser.step()
	return model.eval(), train_hist, test_hist

def format_torch_data():
	x, y = data_preprocessing()

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
	X_train = torch.from_numpy(np.array(x_train)).float()
	y_train = torch.from_numpy(np.array(y_train)).float()
	X_test = torch.from_numpy(np.array(x_test)).float()
	y_test = torch.from_numpy(np.array(y_test)).float()

	return X_train, y_train, X_test, y_test

def covid_model_train(seq_length=42, hidden_dim=10, n_layer=2, lr=0.01):
	X_train, y_train, X_test, y_test = format_torch_data()

	model = Covid_LSTM(
		n_features=1,
		n_hidden=hidden_dim,
		seq_len=seq_length,
		n_layers=n_layer
	)
	model, train_hist, test_hist = train_model(
		model,
		X_train,
		y_train,
		X_test,
		y_test,
		lr=lr
	)

def main():
	covid_model_train(seq_length=42, hidden_dim=10, n_layer=2, lr=0.01)

if __name__ == "__main__":
	main()
