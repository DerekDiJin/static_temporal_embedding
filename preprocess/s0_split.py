import os,sys
import datetime
from pathlib import Path
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
import matplotlib
import matplotlib.pyplot as plt

import scipy
import scipy.sparse as sps
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds, eigs
import sparsesvd

import sklearn
from sklearn import metrics
# from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import validation_curve
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# from sklearn import svm
# from sklearn import preprocessing
from sklearn.decomposition import NMF, DictionaryLearning
from sklearn.manifold import TSNE

from datetime import datetime
from collections import defaultdict

	
# def temp_split(input_temp_file_path, percentage_testing):

# 	print input_temp_file_path
# 	prefix, ext = os.path.splitext(input_temp_file_path)
# 	output_file_path_train = prefix + '_train.tsv'
# 	output_file_path_test = prefix + '_test.tsv'


# 	###############################################################
# 	# Create the train & test splits
# 	###############################################################

# 	ET_sorted = []
# 	max_id = 0

# 	fIn = open(input_temp_file_path, 'r')
# 	for line in fIn.readlines():

# 		(src_str, dst_str, wei_str, temp_str) = line.split('\t')
# 		src = int(src_str)
# 		dst = int(dst_str)
# 		wei = int(wei_str)
# 		temp = int(temp_str)

# 		if src == dst:
# 			continue

# 		temp_edge = (src, dst, wei, temp)

# 		ET_sorted.append(temp_edge)
# 		max_id = max_id if max_id > src else src
# 		max_id = max_id if max_id > dst else dst

# 	fIn.close()

# 	ET_sorted.sort(key=lambda k:k[3])
# 	ET_train = []; ET_test = []

# 	t_init = ET_sorted[0][3]
# 	t_end = ET_sorted[-1][3]
# 	duration = t_end - t_init

# 	# t_split = t_init + duration * percentage_training

# 	n_split = int(len(ET_sorted) * (1 - percentage_testing))
# 	n_test = min( int(len(ET_sorted) * (percentage_testing)), 100000)

# 	idx_split = n_split - 1
# 	idx_init = idx_split - n_test
# 	idx_end = idx_split + n_test

# 	print t_init, t_end
# 	print '[ET_starting] ' + str(datetime.fromtimestamp(t_init))
# 	print '[ET_ending] ' + str(datetime.fromtimestamp(t_end))
# 	print idx_init, idx_split, len(ET_sorted)
# 	print ET_sorted[idx_init][3]

# 	edge_dict = defaultdict(set)

# 	fOut_train = open(output_file_path_train, 'w')
# 	fOut_test = open(output_file_path_test, 'w')

# 	for idx, (src, dst, wei, time) in enumerate(ET_sorted):

# 		edge_dict[src].add(dst)

# 		new_line = str(src) + '\t' + str(dst) + '\t' + str(wei) + '\t' + str(time) + '\n'

# 		if idx < idx_split and idx >= ( idx_init ):
# 			fOut_train.write(new_line)
# 			ET_train.append((src, dst, wei, time))
# 		elif idx < idx_end and idx >= idx_split:
# 			fOut_test.write(new_line)
# 			ET_test.append((src, dst, wei, time))
# 		else:
# 			continue

# 	print '[ET_train size] ' + str(len(ET_train))
# 	print 'ET_test size ' + str(len(ET_test))

# 	print '-----------'

# 	print '[ET_train starting] ' + str(datetime.fromtimestamp(ET_sorted[idx_init][3]))
# 	print '[ET_train ending] ' + str(datetime.fromtimestamp(ET_sorted[idx_split][3]))

# 	print '[ET_test starting] ' + str(datetime.fromtimestamp(ET_sorted[idx_split][3]))
# 	print '[ET_test ending] ' + str(datetime.fromtimestamp(ET_sorted[idx_end][3]))
	
# 	fOut = open('test.tsv', 'w')

# 	for ele in ET_sorted[idx_init:idx_split]:
# 		fOut.write( str(datetime.fromtimestamp(ele[3])) + '\n' )

# 	fOut.write('------')

# 	for ele in ET_sorted[idx_split:idx_end]:
# 		fOut.write( str(datetime.fromtimestamp(ele[3])) + '\n' )

# 	fOut.close()

# 	# for (src, dst, wei, time) in ET_sorted:

# 	# 	new_line = str(src) + '\t' + str(dst) + '\t' + str(wei) + '\t' + str(time) + '\n'

# 	# 	if time <= t_split:
# 	# 		fOut_train.write(new_line)
# 	# 	else:
# 	# 		fOut_test.write(new_line)

# 	fOut_train.close()
# 	fOut_test.close()

# 	###############################################################
# 	# Create the negative samples
# 	###############################################################

# 	k = len(ET_train) * 2
# 	print '[k value] ' + str(k)

# 	fake_edges = np.zeros([k, 2], dtype=int)
# 	unique_edges = set([])

# 	counter = 0
# 	while counter < k:
# 		if counter % 10000 == 0:
# 			print '[generate fake edges] ' + str(counter)

# 		random_src, random_dst = random.sample(range(max_id),2)

# 		if (random_src in edge_dict and random_dst in edge_dict[random_src]) or ((random_src, random_dst) in unique_edges):
# 			continue
# 		else:
# 			unique_edges.add((random_src, random_dst))
# 			fake_edges[counter,:] = [random_src, random_dst]
# 			counter += 1

# 	N_train = len(ET_train)

# 	fake_edges_train = fake_edges[:N_train,:]
# 	fake_edges_test = fake_edges[N_train:,:]

# 	np.savetxt(input_temp_file_path.split('.')[0] + '_fake_edges_train.tsv', fake_edges_train, fmt='%i',  delimiter='\t')
# 	np.savetxt(input_temp_file_path.split('.')[0] + '_fake_edges_test.tsv', fake_edges_test, fmt='%i', delimiter='\t')



# 	return


def exp_generate(input_temp_files_dir, training_init, training_end, testing_init, testing_end):

	print '[input_temp_files_dir] ' + input_temp_files_dir

	prefix = input_temp_files_dir[:-1]
	output_file_path_train = prefix + '_train.tsv'
	output_file_path_test = prefix + '_test.tsv'

	files_all = os.listdir(input_temp_files_dir.replace('\ ', ' '))
	print files_all

	ET_train_all = []; ET_test = []; nodes_train_all = set([])

	for idx in range(training_init, training_end+1):

		cur_edges = []

		cur_input_file = input_temp_files_dir + str(idx) + '.tsv'
		fIn = open(cur_input_file, 'r')
		lines = fIn.readlines()
		fIn.close()

		ET_train_all += lines

		for line in lines:
			src_str, dst_str, wei_str, time_str = line.strip().split('\t')
			nodes_train_all.add(int(src_str))
			nodes_train_all.add(int(dst_str))


	for idx in range(testing_init, testing_end+1):

		cur_input_file = input_temp_files_dir + str(idx) + '.tsv'
		fIn = open(cur_input_file, 'r')
		lines = fIn.readlines()
		fIn.close()

		for line in lines:
			src_str, dst_str, wei_str, time_str = line.strip().split('\t')
			# if int(src_str) in nodes_train_all and int(dst_str) in nodes_train_all:

			ET_test.append(line)

		


	edge_dict_train = defaultdict(set); edge_dict_test = defaultdict(set)
	k_train = len(ET_test) * 1
	k_test = len(ET_test)
	ET_train = random.sample(ET_train_all, k_train)
	max_id_train = 0; max_id_test = 0

	fOut_train = open(output_file_path_train, 'w')
	fOut_test = open(output_file_path_test, 'w')

	for line in ET_train:
		src_str, dst_str, wei_str, time_str = line.strip().split('\t')
		fOut_train.write(line)
		max_id_train = max(max_id_train, int(src_str))
		max_id_train = max(max_id_train, int(dst_str))

		edge_dict_train[int(src_str)].add(int(dst_str))

	for line in ET_test:
		src_str, dst_str, wei_str, time_str = line.strip().split('\t')
		fOut_test.write(line)
		max_id_test = max(max_id_test, int(src_str))
		max_id_test = max(max_id_test, int(dst_str))

		edge_dict_test[int(src_str)].add(int(dst_str))

	fOut_train.close()
	fOut_test.close()

	###############################################


	fake_edges_train = np.zeros([k_train, 2], dtype=int); fake_edges_test = np.zeros([k_test, 2], dtype=int)
	unique_edges_train = set([]); unique_edges_test = set([])

	counter = 0
	while counter < k_train:
		if counter % 10000 == 0:
			print '[generate fake edges] ' + str(counter)

		random_src, random_dst = random.sample(range(max_id_train),2)

		if (random_src in unique_edges_train and random_dst in unique_edges_train[random_src]) or ((random_src, random_dst) in unique_edges_train):
			continue
		else:
			unique_edges_train.add((random_src, random_dst))
			fake_edges_train[counter,:] = [random_src, random_dst]
			counter += 1

	counter = 0
	while counter < k_test:
		if counter % 10000 == 0:
			print '[generate fake edges] ' + str(counter)

		random_src, random_dst = random.sample(range(max_id_test),2)

		if (random_src in unique_edges_test and random_dst in unique_edges_test[random_src]) or ((random_src, random_dst) in unique_edges_test):
			continue
		else:
			unique_edges_test.add((random_src, random_dst))
			fake_edges_test[counter,:] = [random_src, random_dst]
			counter += 1



	np.savetxt(prefix + '_fake_edges_train.tsv', fake_edges_train, fmt='%i',  delimiter='\t')
	np.savetxt(prefix + '_fake_edges_test.tsv', fake_edges_test, fmt='%i', delimiter='\t')




	




if __name__ == '__main__':
	
	if len(sys.argv) != 4:
		sys.exit('usage: split.py <input_temp_files_dir> <#snapshot_training:[init,end]> <#snapshot_testing[init,end]>')

	input_temp_files_dir = sys.argv[1]
	snapshots_training_init, snapshots_training_end = sys.argv[2].split(',')
	snapshots_testing_init, snapshots_testing_end = sys.argv[3].split(',')

	# temp_split(input_temp_file_path, percentage_testing)

	exp_generate(input_temp_files_dir, int(snapshots_training_init), int(snapshots_training_end), int(snapshots_testing_init), int(snapshots_testing_end))
	




