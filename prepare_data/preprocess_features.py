import os
import sys
import json
import numpy as np

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
sys.path.insert(0, PROJECT_DIR)

from definitions import constants
DATA_DIR = constants["DATA_DIR"]


def get_deltas(features):
	last_feature = np.array(features[0])
	last_delta = np.array([0]*len(features[0]))
	deltas = []
	delta_deltas = []
	# append 0 vectors to delta, delta deltas at beginning of sequence
	deltas.append(np.array([0]*len(features[0])))
	delta_deltas.append(np.array([0]*len(features[0])))
	delta_deltas.append(np.array([0]*len(features[0])))
	
	for i in range(1, len(features)):	
		current = np.array(features[i])
		delta = current - last_feature
		deltas.append(delta)
		if i >= 2:
			delta_delta = delta - last_delta
			delta_deltas.append(delta_delta)
		last = current
		last_delta = delta

	features = np.array(features)
	deltas = np.array(deltas)
	delta_deltas = np.array(delta_deltas)
	concatenated = np.concatenate([features, deltas, delta_deltas], axis=1)
	return concatenated

# with open(os.path.join(DATA_DIR, 'train.json')) as f:
# 	train = json.load(f)
# print('train loaded')

with open(os.path.join(DATA_DIR, 'dev.json')) as f:
	dev = json.load(f)
print('dev loaded')

train_non_padded = []

for datapoint in train:
	features = train[datapoint]['features']
	train_non_padded.append(get_deltas(features))
	print(train_array[-1].shape)

dev_non_padded = []

for datapoint in dev:
	features = dev[datapoint]['features']
	dev_non_padded.append(get_deltas(features))
	
max_length = max([len(datapoint) for datapoint in train_non_padded + dev_non_padded])

dev_padded = []
for datapoint in dev_non_padded:
	padded_datapoint = np.zeros(shape=(max_length, dev_non_padded[0].shape[1]))
	padded_datapoint[:datapoint.shape[0], :datapoint.shape[1]] = datapoint
	dev_padded.append(np.expand_dims(padded_datapoint, axis=2))
dev_padded = np.concatenate(dev_padded, axis=2)

train_padded = []
for datapoint in train_non_padded:
	padded_datapoint = np.zeros(shape=(max_length, dev_non_padded[0].shape[1]))
	padded_datapoint[:datapoint.shape[0], :datapoint.shape[1]] = datapoint
	train_padded.append(np.expand_dims(padded_datapoint, axis=2))
train_padded = np.concatenate(train_padded, axis=2)

np.save('data/train.npy', train_padded)
np.save('data/dev.npy', dev_padded)

	
