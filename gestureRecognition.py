import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import pandas as pd
import os
import glob
from sklearn import decomposition
from sklearn.decomposition import RandomizedPCA
from scipy.fftpack import dct
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import cPickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Import multiple csv files into pandas and concatenate into one DataFrame
def load_data():
	path = os.path.join(os.path.dirname(__file__), 'Gesture Data')
	allFiles = glob.glob(path + "/*.txt")
	data = pd.DataFrame()
	list_ = []
	for idx, file_ in enumerate(allFiles):
		df = pd.read_csv(file_,index_col=None, header=0)
		np_df = df.transpose().as_matrix()
		input_pattern_frequency = np.array([])
		for elm in np_df:
			channel_in_frequency = np.fft.rfft(elm).real
			input_pattern_frequency = np.concatenate([channel_in_frequency, input_pattern_frequency])
		input_pattern_frequency = input_pattern_frequency.reshape(1,input_pattern_frequency.size)
		frame = pd.DataFrame.from_records(input_pattern_frequency)
		data = data.append(frame)
		list_.append(int(file_[20]));
	data['class'] = list_
	data.to_csv('example.csv', index=False)

def main():

	var1 = sys.argv[1]

	if(var1 == 'train'):
		# Use this function only once
		load_data()
		# Read data from .csv file
		data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'example.csv'),index_col=None, header=0)
		# Usually we assign missing values using the mean of each col from the training data(for numeric attributes), since missing points were not above 10% of data, I just decided to assign zero to them, to simplify things (I took a risk of building a biased model doing so).
		data = data.apply(lambda x: x.fillna(0),axis=0)

		# Generate the training set.  Set random_state to be able to replicate results.
		# Select anything not in the training set and put it in the testing set.
		msk = np.random.rand(len(data)) < 0.8
		temp_train = data[msk]
		test = data[~msk]

		temp = np.random.rand(len(temp_train)) < 0.8
		train = temp_train[temp]
		validation = temp_train[~temp]
		#print data
		print "Information on dataset"
		print "train", train.shape
		print "valid", validation.shape
		print "test", test.shape
		test.to_csv("test_dataset.csv", sep=',',  index=False)
		print "Running the model"

		model = RandomForestClassifier(n_estimators=100)

		# Fit the model to the data.
		model.fit(train[train.columns.difference(['class'])], train["class"])

		# Make predictions.
		predictions = model.predict(validation[validation.columns.difference(['class'])])

		# Compute the error.
		accuracy = accuracy_score(validation["class"], predictions, normalize=False) / float(validation["class"].size)
		error_rate = 1 - accuracy
		print error_rate

		with open('RandomForestClassifier', 'wb') as f:
			cPickle.dump(model, f)
	else:
		data = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_dataset.csv"), delimiter=',', encoding="utf-8-sig")
		ground_truth = data["class"].as_matrix()
		with open('RandomForestClassifier', 'rb') as f:
			rf = cPickle.load(f)
			predictions = rf.predict(data[data.columns.difference(["class"])])
			predictions.tofile("results.csv", sep='\n', format='%.1f')
			ground_truth.tofile("ground_truth.csv", sep='\n', format='%.1f')
			df = pd.DataFrame({'predictions': predictions, 'ground_truth': ground_truth})
			print predictions
			print ground_truth
			sns.set(style="darkgrid")
			sns.regplot(x="ground_truth", y='predictions', data=df);
			sns.plt.show()
			# Compute the error.
			accuracy = accuracy_score(ground_truth, predictions, normalize=False) / float(ground_truth.size)
			error_rate = 1 - accuracy
			print "Test Error:"
			print error_rate

if __name__ == "__main__":
	main()
