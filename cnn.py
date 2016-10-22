import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
import pandas as pd
import os
import glob
from sklearn import decomposition
from sklearn.decomposition import RandomizedPCA
from scipy.fftpack import dct

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

# Import multiple csv files into pandas and concatenate into one DataFrame
def load_data():
	path = os.path.join(os.path.dirname(__file__), 'Gesture Data')
	allFiles = glob.glob(path + "/*.txt")
	data = pd.DataFrame()
	#data = np.array([]).reshape(0,200)
	list_ = []
	for file_ in allFiles:
		df = pd.read_csv(file_,index_col=None, header=0)
		np_df = df.transpose().as_matrix()
		input_pattern_frequency = np.array([])
		for elm in np_df:
			channel_in_frequency = np.fft.rfft(elm).real
			input_pattern_frequency = np.concatenate([channel_in_frequency, input_pattern_frequency])
		input_pattern_frequency = input_pattern_frequency.reshape(1,input_pattern_frequency.size)
		frame = pd.DataFrame.from_records(input_pattern_frequency)
		data = data.append(frame)
		print data.shape
	#frame = pd.concat(list_)
	#input_pattern_frequency_col_mean = input_pattern_frequency.mean(axis=0)
	#np.savetxt('example.csv', data, delimiter=',')
	#data.to_csv('example.csv')
	#print frame

if __name__=="__main__":
	load_data()
	#data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'example.csv'),index_col=None, header=0)
	#print data.shape

