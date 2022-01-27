import sys
import datetime
from pathlib import Path
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
# import matplotlib
# import matplotlib.pyplot as plt

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


def read_in_comm_format(emb_file, delimiter):
	representations_dict = {}
	fIn = open(emb_file, 'r')
	lines = fIn.readlines()
	for line in lines:
		parts = line.strip('\r\n').split('\t')
		node_id = int(parts[0])
		comm_id = int(parts[1])

		representations_dict[node_id] = comm_id

	fIn.close()
	return representations_dict

def read_in_adobe_data(emb_file, nodes_to_read, delimiter):

	print '----'
	representations_dict = {}

	fIn = open(emb_file, 'r')
	for line in fIn.readlines():
		if len(line.strip().split(' ')) == 2:
			continue
		else:
			parts = line.strip().split(' ')
			key = int(parts[0])
			if key in nodes_to_read:
				value = [float(ele) for ele in parts[1:]]
				representations_dict[key] = value

	return representations_dict

	# representation_unorder = np.genfromtxt(emb_file, dtype=float, delimiter=delimiter, skip_header=1)
	# m, n = representation_unorder.shape
	# print 'representation_unorder read in.'
	# for i in range(m):
	# 	if i % 50000 == 0:
	# 		print i
	# 	key = int(representation_unorder[i, 0])
	# 	value = representation_unorder[i, 1:]
	# 	representations_dict[key] = value

	# representations = np.zeros([max(representations_dict,keys())+1, n-1])
	# for key in representations_dict.keys():
	# 	# print key
	# 	representations[key,:] = representations_dict[key]

	# return representations_dict


def read_in_node2vec_format(emb_file, delimiter):

	print('----')
	representations_dict = {}
	representation_unorder = np.genfromtxt(emb_file, dtype=float, delimiter=delimiter, skip_header=1)
	m, n = representation_unorder.shape
	print 'representation_unorder read in.'
	for i in range(m):
		if i % 50000 == 0:
			print i
		key = int(representation_unorder[i, 0])
		value = representation_unorder[i, 1:]
		representations_dict[key] = value

	# representations = np.zeros([max(representations_dict,keys())+1, n-1])
	# for key in representations_dict.keys():
	# 	# print key
	# 	representations[key,:] = representations_dict[key]

	return representations_dict



def find_ones_edges(raw, k):

	E = raw.shape[0]
	train_ones_idx = random.sample(range(E), k)

	 # = ones_idx
	# test_ones_idx = ones_idx[k:]

	actual_edges_train = raw[train_ones_idx, 0:2].tolist()
	# actual_edges_test = raw[test_ones_idx, 0:2].tolist()

	actual_edges = actual_edges_train

	return actual_edges


def find_zeros_edges(raw, k):

	edge_dict = {}
	E = raw.shape[0]
	edge_list = raw[:,0:2].tolist()
	max_src = max(raw[:,0:2][:,0])
	max_dst = max(raw[:,0:2][:,1])
	max_id = max(max_src, max_dst)


	counter = 0

	fake_edges = np.zeros([k*2, 2], dtype=int)

	while counter < k*2:
		print '[generate fake edges] ' + str(counter)
		random_src = random.randint(0, max_id)
		random_dst = random.randint(0, max_id)
		# random_edge = random.sample(range(max_id), 2)
		if random_dst not in edge_dict[random_src]:

			fake_edges[counter,:] = [random_src, random_dst]
			counter += 1
			if counter % 1000 == 0:
				print counter


	fake_edges_train = fake_edges[:k,:]
	fake_edges_test = fake_edges[k:,:]

	fake_edges = [fake_edges_train, fake_edges_test]

	return fake_edges


def compute_edge_features(edges, representations, method):

	n = len(representations)
	f = len(representations[next(iter(representations))])#len(representations[10]) #669078

	if 'concat' in method:
		X_e = np.zeros((len(edges), f*2))
	else:
		X_e = np.zeros((len(edges), f))

	print "[compute_edge_features]  n="+str(n) + ", f=" + str(f) + ", method=" + method

	counter = 0
	for edge in edges:
		src = edge[0]
		dst = edge[1]

		x = representations[src] if src in representations else [0.0] * f
		y = representations[dst] if dst in representations else [0.0] * f

		# print x
		# print y
		if counter % 10000 == 0:
			print "[compute_edge_features] " + str(counter)
		val = np.array([])

		if 'mean' in method:
			val = ((x+y) * 1.0)/2.0
		elif 'prod' in method:
			val = x * y
		elif 'weighted-L1' in method:
			val = abs(x-y)
		elif 'weighted-L2' in method:
			val = abs(x-y)*abs(x-y)
		elif 'concat' in method:
			val = np.concatenate((x,y))
		elif 'sum' in method:
			val = x+y
		elif 'max' in method:
			val = np.max((x,y),axis=0)
		elif 'diff' in method:
			val = x-y
		else:
			sys.exit('Operation not supported.')

		# print (x, y)
		X_e[counter,:] = val
		counter += 1

	return X_e


def read_in_gs_format(input_emb_file_path, delimiter):

	representations = np.genfromtxt(input_emb_file_path, dtype=float, delimiter=delimiter)

	m, n = representations.shape
	representation_dict = {}

	for i in range(m):
		representation_dict[i] = representations[i,:]

	return representation_dict

def read_in_npz_format(input_emb_file_path, delimiter):


	representations = np.load(input_emb_file_path, allow_pickle=True, encoding="latin1")['data']

	m, n = representations.shape
	representation_dict = {}

	for i in range(m):
		representation_dict[i] = representations[i,:]

	return representation_dict



def read_in_nodes_splitted_er(nodes_splitted_file_path):
	result = {}
	fIn = open(nodes_splitted_file_path, 'r')

	lines = fIn.readlines()
	for line in lines:
		orig_id, new_id = line.strip('\r\n').split('\t')
		result[int(orig_id)] = int(new_id)
		#result.append( int(line.strip('\r\n')) )
	# print result
	fIn.close()
	return result

def read_er_result(input_emb_file_path):
	result = set()
	fIn = open(input_emb_file_path, 'r')

	lines = fIn.readlines()
	for line in lines:
		src, dst = line.strip('\r\n').split('\t')

		result.add((int(src), int(dst)))
		result.add((int(dst), int(src)))

	fIn.close()

	return result



def read_in_nodes_splitted(nodes_splitted_file_path, representations_dict):
	result = {}
	fIn = open(nodes_splitted_file_path, 'r')

	lines = fIn.readlines()
	for line in lines:
		orig_id, new_id = line.strip('\r\n').split('\t')
		if (int(new_id) in representations_dict) and (int(orig_id) in representations_dict):
			result[int(orig_id)] = int(new_id)
		#result.append( int(line.strip('\r\n')) )
	# print result
	fIn.close()
	return result


def metrics_output(y_true_a, y_pred, y_true_b, y_score):
	AUC = metrics.roc_auc_score(y_true=y_true_b, y_score=y_score)
	ACC = metrics.accuracy_score(y_true_a, y_pred)
	P = metrics.precision_score(y_true_a, y_pred)
	R = metrics.recall_score(y_true_a, y_pred)

	f1_micro = metrics.f1_score(y_true_a, y_pred, average='micro')
	f1_macro = metrics.f1_score(y_true_a, y_pred, average='macro')

	# score = clf_t.score(X_test, Y_test)
	print 'AUC, ACC, P, R:'
	print(AUC, ACC, P, R)
	print 'f1_micro & f1_macro:'
	print(f1_micro, f1_macro)

	return AUC, ACC, f1_macro


def read_in_n2b(input_table_path):
	result = {}

	fIn = open(input_table_path, 'r')
	lines = fIn.readlines()

	for line in lines:
		signature, nodes_str = line.strip('\r\n').split('\t')

		nodes = nodes_str.strip('[]').split(',')

		for node in nodes:
			result[int(node)] = signature



	return result

def get_nodes_to_read(input_real_train_edge_file_path, input_fake_train_edge_file_path, input_real_test_edge_file_path, input_fake_test_edge_file_path):

	files = [input_real_train_edge_file_path, input_fake_train_edge_file_path, input_real_test_edge_file_path, input_fake_test_edge_file_path]

	result = set([])
	for cur_file in files:
		fIn = open(cur_file, 'r')

		for line in fIn.readlines():
			src, dst = line.strip().split('\t')
			result.add(int(src)); result.add(int(dst))

		fIn.close()

	print ':)'
	print len(result)

	return result




def eval(argv):

	############ TODO: adding weights in labeling? ############

	# k = 1000
	method = 'concat' #'mean'

	iput_emb_file_path = argv[1]
	print iput_emb_file_path

	input_real_train_edge_file_path = argv[2]
	print input_real_train_edge_file_path

	input_fake_train_edge_file_path = argv[3]
	print input_fake_train_edge_file_path

	input_real_test_edge_file_path = argv[4]
	print input_real_test_edge_file_path

	input_fake_test_edge_file_path = argv[5]
	print input_fake_test_edge_file_path

	mod = argv[6]

	delimiter = " "
	if ".csv" in input_real_train_edge_file_path:
		delimiter = ","
	elif ".tsv" in input_real_train_edge_file_path:
		delimiter = "\t"
	elif ".npz" in input_real_train_edge_file_path:
		delimiter = " "
	else:
		sys.exit('Format not supported.')



	if mod == 'gs':
		# representations = np.genfromtxt(iput_host_file_path, dtype=float, delimiter=delimiter)

		representations = read_in_gs_format(iput_emb_file_path, ' ')


	elif mod == 'node2vec':
		representations = read_in_node2vec_format(iput_emb_file_path, ' ')
		# nodes_to_read = get_nodes_to_read(input_real_train_edge_file_path, input_fake_train_edge_file_path, input_real_test_edge_file_path, input_fake_test_edge_file_path)
		# representations = read_in_adobe_data(iput_emb_file_path, nodes_to_read, ' ')
		# print representations[0]

	elif mod == 'npz':
		representations = read_in_npz_format(iput_emb_file_path, ' ')

	else:
		sys.exit('mode not supported.')

		
	train_ones_edges = np.genfromtxt(input_real_train_edge_file_path, usecols=(0,1), dtype=int, delimiter=delimiter)
	train_zero_edges = np.genfromtxt(input_fake_train_edge_file_path, usecols=(0,1), dtype=int, delimiter=delimiter)

	test_zero_edges = np.genfromtxt(input_fake_test_edge_file_path, usecols=(0,1), dtype=int, delimiter=delimiter)
	test_ones_edges = np.genfromtxt(input_real_test_edge_file_path, usecols=(0,1), dtype=int, delimiter=delimiter)

	X_1_train = compute_edge_features(train_ones_edges, representations, method)
	X_1_test = compute_edge_features(test_ones_edges, representations, method)

	X_0_train = compute_edge_features(train_zero_edges, representations, method)
	X_0_test = compute_edge_features(test_zero_edges, representations, method)

	print '-------------'
	# print representations.shape

	train_N = train_ones_edges.shape[0]
	test_N = test_ones_edges.shape[0]

	train_ones = np.ones([train_N, 1], dtype=int)
	train_zeros = np.zeros([train_N, 1], dtype=int)
	test_ones = np.ones([test_N, 1], dtype=int)
	test_zeros = np.zeros([test_N, 1], dtype=int)


	X_train = np.vstack((X_1_train, X_0_train))
	X_test = np.vstack((X_1_test, X_0_test))

	Y_train = np.vstack((train_ones, train_zeros))

	##########################################
	# One column
	#########################################
	Y_test_a = np.vstack((test_ones, test_zeros))

	##########################################
	# Two columns
	#########################################
	Y_test_0 = np.vstack((test_zeros, test_ones))
	Y_test_1 = np.vstack((test_ones, test_zeros))
	Y_test_b = np.hstack((Y_test_0, Y_test_1))

	print '-------'

	clf_t = LogisticRegression(C = 1.0, tol=1e-4)
	clf_t.fit(X_train, Y_train)

	y_pred_prob = clf_t.predict_proba(X_test)
	print y_pred_prob
	y_pred = clf_t.predict(X_test)
	print y_pred

	AUC = metrics.roc_auc_score(y_true=Y_test_b, y_score=y_pred_prob)
	ACC = metrics.accuracy_score(Y_test_a, y_pred)
	P = metrics.precision_score(Y_test_a, y_pred)
	R = metrics.recall_score(Y_test_a, y_pred)

	f1_micro = metrics.f1_score(Y_test_a, y_pred, average='micro')
	f1_macro = metrics.f1_score(Y_test_a, y_pred, average='macro')

	# score = clf_t.score(X_test, Y_test)
	print 'AUC, ACC, P, R:'
	print(AUC, ACC, P, R)
	print 'f1_micro & f1_macro:'
	print(f1_micro, f1_macro)

	return AUC, ACC, f1_macro

		

	

	# if mod != 'er' and mod != 'gser':

	# 	nodes_splitted = read_in_nodes_splitted(nodes_splitted_file_path, representations_dict)
	# 	# common_keys = set(representations_dict.keys()).intersection( set(representations_dict_2.keys()) )
		
	# 	nodes_splitted_list = list(nodes_splitted.keys())
	# 	# print nodes_splitted_list
	# 	k = len(nodes_splitted_list)
	# 	print 'k:' + str(k)

	# 	train_one_idx = nodes_splitted_list[:k/2]
	# 	test_one_idx = nodes_splitted_list[k/2:]

	# 	X_one_train = get_edge_features(representations_dict, nodes_splitted, train_one_idx, method, True)
	# 	X_one_test = get_edge_features(representations_dict, nodes_splitted, test_one_idx, method, True)

	# 	X_zero_train = get_edge_features(representations_dict, nodes_splitted, train_one_idx, method, False)
	# 	X_zero_test = get_edge_features(representations_dict, nodes_splitted, test_one_idx, method, False)



	# 	ones_train = np.ones([len(train_one_idx), 1], dtype=int)
	# 	zeros_train = np.zeros([len(train_one_idx), 1], dtype=int)

	# 	ones_test = np.ones([len(test_one_idx), 1], dtype=int)
	# 	zeros_test = np.zeros([len(test_one_idx), 1], dtype=int)


	# 	X_train = np.vstack((X_one_train, X_zero_train))
	# 	X_test = np.vstack((X_one_test, X_zero_test))

	# 	Y_train = np.vstack((ones_train, zeros_train))

	# 	##########################################
	# 	# One column
	# 	#########################################
	# 	Y_test_a = np.vstack((ones_test, zeros_test))

	# 	##########################################
	# 	# Two columns
	# 	#########################################
	# 	Y_test_0 = np.vstack((zeros_test, ones_test))
	# 	Y_test_1 = np.vstack((ones_test, zeros_test))
	# 	Y_test_b = np.hstack((Y_test_0, Y_test_1))

	# 	print '-------'
	# 	# print X_train
	# 	# print X_test
	# 	# print Y_test.shape

	
		
	# 	clf_t = LogisticRegression(C = 1.0, tol=1e-4)
	# 	clf_t.fit(X_train, Y_train)

	# 	y_pred_prob = clf_t.predict_proba(X_test)
	# 	y_pred = clf_t.predict(X_test)
	# 	# print y_pred_prob
	# 	# print 0 in y_pred
	# 	# print np.where(y_pred == 0)


	# 	AUC = metrics.roc_auc_score(y_true=Y_test_b, y_score=y_pred_prob)
	# 	ACC = metrics.accuracy_score(Y_test_a, y_pred)
	# 	P = metrics.precision_score(Y_test_a, y_pred)
	# 	R = metrics.recall_score(Y_test_a, y_pred)

	# 	f1_micro = metrics.f1_score(Y_test_a, y_pred, average='micro')
	# 	f1_macro = metrics.f1_score(Y_test_a, y_pred, average='macro')

	# 	# score = clf_t.score(X_test, Y_test)
	# 	print 'AUC, ACC, P, R:'
	# 	print(AUC, ACC, P, R)
	# 	print 'f1_micro & f1_macro:'
	# 	print(f1_micro, f1_macro)


	# 	return AUC, ACC, f1_macro

	# else:
	# 	print 'TODO'

		# <TODO>
		# test_zero_edges = np.genfromtxt(input_fake_test_edge_file_path, dtype=int, delimiter=delimiter)
		# test_ones_edges = np.genfromtxt(input_real_test_edge_file_path, dtype=int, delimiter=delimiter)

		# y_pred = np.empty([test_zero_edges.shape[0] + test_ones_edges.shape[0], 1])

		# for i in range(test_zero_edges.shape[0]):
		# 	if representations[test_zero_edges[i,0]] != representations[test_zero_edges[i,1]]:
		# 		y_pred[i, 0] = 0
		# 	else:
		# 		y_pred[i, 0] = 1

		# for i in range(test_ones_edges.shape[0]):
		# 	if representations[test_ones_edges[i,0]] == representations[test_ones_edges[i,1]]:
		# 		y_pred[i+test_zero_edges.shape[0], 0] = 1
		# 	else:
		# 		y_pred[i+test_zero_edges.shape[0], 0] = 0

		# y_pred_prob = y_pred

		# ones = np.ones([test_ones_edges.shape[0], 1], dtype=int)
		# zeros = np.zeros([test_ones_edges.shape[0], 1], dtype=int)

		# ##########################################
		# # One column
		# #########################################
		# Y_test_a = np.vstack((ones, zeros))

		# ##########################################
		# # Two columns
		# #########################################
		# Y_test_0 = np.vstack((zeros, ones))
		# Y_test_1 = np.vstack((ones, zeros))
		# Y_test_b = np.hstack((Y_test_0, Y_test_1))


		


	# clf_t.predict(X_test)
	# roc_auc_score()
	# sklearn.metrics.auc()
	# AUC = metrics.roc_auc_score(y_true=Y_test_b, y_score=y_pred_prob)
	# ACC = metrics.accuracy_score(Y_test_a, y_pred)
	# P = metrics.precision_score(Y_test_a, y_pred)
	# R = metrics.recall_score(Y_test_a, y_pred)

	# f1_micro = metrics.f1_score(Y_test_a, y_pred, average='micro')
	# f1_macro = metrics.f1_score(Y_test_a, y_pred, average='macro')

	# # score = clf_t.score(X_test, Y_test)
	# print 'AUC, ACC, P, R:'
	# print(AUC, ACC, P, R)
	# print 'f1_micro & f1_macro:'
	# print(f1_micro, f1_macro)
	# print representations[0,:]


	return




if __name__ == '__main__':
	
	if len(sys.argv) != 7:
		sys.exit('usage: stats_edges.py <input_emb_file_path> <input_real_train_graph_file_path> <input_fake_train_edge_file_path> <input_real_test_edge_file_path> <input_fake_test_edge_file_path> <mod>')

	multi = False

	if multi:
		AUCs=[]; ACCs=[]; F1s = []
		for _ in range(10):
			AUC, ACC, F1 = eval(sys.argv)
			AUCs.append(AUC); ACCs.append(ACC); F1s.append(F1)

		print 'AUC: ', 'mean: ', np.mean(AUCs), 'std: ', np.std(AUCs)
		print 'ACC: ', 'mean: ', np.mean(ACCs), 'std: ', np.std(ACCs)
		print 'F1: ', 'mean: ', np.mean(F1s), 'std: ', np.std(F1s)

	else:
		AUC, ACC, F1 = eval(sys.argv)
		print '?-+', round(AUC, 4), round(ACC, 4), round(F1, 4)




