import sys
import datetime
from pathlib import Path
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
import pickle

# import matplotlib
# import matplotlib.pyplot as plt

import scipy.sparse as sps
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds, eigs
import sparsesvd

# import sklearn
# from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import validation_curve
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# from sklearn import svm
# from sklearn import preprocessing
from sklearn.decomposition import NMF, DictionaryLearning
from sklearn.manifold import TSNE

from collections import defaultdict


def construct_cat(input_gt_path, delimiter):
	
	####################################################
	# Input: per line, 1) cat-id_init, id_end or 2) cat-id
	#
	# Return: 1) dict: cat-ids and 2) id-cat
	####################################################

	result = defaultdict(set)
	id_cat_dict = dict()

	fIn = open(input_gt_path, 'r')
	lines = fIn.readlines()
	for line in lines:

		parts = line.strip('\r\n').split(delimiter)
		if len(parts) == 3:
			cat = parts[0]
			node_id_start = parts[1]
			node_id_end = parts[2]

			for i in range( int(node_id_start), int(node_id_end)+1 ):
				result[ int(cat) ].add( i )
				id_cat_dict[i] = int(cat)

		elif len(parts) == 2:
			cat = parts[0]
			node_id = parts[1]

			result[int(cat)].add( int(node_id) )
			id_cat_dict[int(node_id)] = int(cat)



		else:
			sys.exit('Cat file format not supported')

	fIn.close()
	return result, id_cat_dict


if __name__ == '__main__':
	
	if len(sys.argv) != 2:
		sys.exit('usage: stats_edges.py <temporal_file_path>')

	random.seed(17)
	# assume the graph is undirected, unweighted
	weighted = True
	directed = True

	# input_file_path = sys.argv[1]
	# cur_file_path = str(Path().resolve().parent)

	input_file_path = sys.argv[1]

	# input_file_path = cur_file_path + '/real_graphs/' + sys.argv[1]
	# input_gt_path = cur_file_path + '/toy_graphs/yahoo-msg-heter_cat.tsv'
	# input_file_path = cur_file_path + '/toy_graphs/bibsonomy_wei.tsv'
	# input_gt_path = cur_file_path + '/toy_graphs/bibsonomy_cat.tsv'
	# input_file_path = cur_file_path + '/toy_graphs/test.tsv'
	# input_gt_path = cur_file_path + '/toy_graphs/test1_cat.tsv'

	##############################
	# for exp
	##############################
	# input_file_path = cur_file_path + '/real_graphs/yahoo-msg-heter/yahoo-msg-heter_wei.tsv'
	# input_gt_path = cur_file_path + '/real_graphs/yahoo-msg-heter/yahoo-msg-heter_cat.tsv'
	# input_file_path = cur_file_path + '/exp/ER_100_500.tsv'
	# input_gt_path = cur_file_path + '/exp/ER_100_500_cat.tsv'
	# input_file_path = cur_file_path + '/exp/ER_1000_5000.tsv'
	# input_gt_path = cur_file_path + '/exp/ER_1000_5000_cat.tsv'
	# input_file_path = cur_file_path + '/exp/ER_10000_50000.tsv'
	# input_gt_path = cur_file_path + '/exp/ER_10000_50000_cat.tsv'
	# input_file_path = cur_file_path + '/exp/ER_100000_500000.tsv'
	# input_gt_path = cur_file_path + '/exp/ER_100000_500000_cat.tsv'
	# input_file_path = cur_file_path + '/exp/ER_1000000_5000000.tsv'
	# input_gt_path = cur_file_path + '/exp/ER_1000000_5000000_cat.tsv'
	# input_file_path = cur_file_path + '/exp/ER_10000000_50000000.tsv'
	# input_gt_path = cur_file_path + '/exp/ER_10000000_50000000_cat.tsv'
	##############################

	delimiter = " "
	if ".csv" in input_file_path:
		delimiter = ","
	elif ".tsv" in input_file_path:
		delimiter = "\t"
	else:
		sys.exit('Format not supported.')

	output_file_path = input_file_path.replace('.tsv', '_base.tsv')
	fOut = open(output_file_path, 'w')

	fIn = open(input_file_path, 'r')
	lines = fIn.readlines()


	for line in lines:
		src, dst, wei, time = line.strip('\r\n').split(delimiter)
		new_line = src + delimiter + dst + delimiter + wei + '\n'
		fOut.write(new_line)

	fIn.close()
	fOut.close()

	# raw = np.genfromtxt(input_file_path, dtype=int)
	# rows = raw[:,0]
	# cols = raw[:,1]
	# weis = raw[:,2]


	# check_eq = True
	# max_id = int(max(max(rows), max(cols)))
	# num_nodes = max_id + 1
	# print '[max_node_id] ' + str(max_id)
	# print '[num_nodes] ' + str(num_nodes)

	# TOTAL = int(proportion * num_nodes)

	# if max(rows) != max(cols):
	# 	rows = np.append(rows,max(max(rows), max(cols)))
	# 	cols = np.append(cols,max(max(rows), max(cols)))
	# 	weis = np.append(weis, 0)
	# 	check_eq = False

	# CAT_DICT, ID_CAT_DICT = construct_cat(input_gt_path, delimiter)

	# adj_matrix = sps.lil_matrix( sps.csc_matrix((np.ones(len(rows)), (rows, cols))))
	# # adj_matrix = sps.lil_matrix( sps.csc_matrix((weis, (rows, cols))))
	# # print adj_matrix.todense()

	# new_id = max_id
	# counter = max_id

	# keys = np.arange(num_nodes)
	# values_out = np.asarray(adj_matrix.sum(1).T)[0]
	# values_in = np.asarray(adj_matrix.sum(0))[0]

	# dict_outdeg = dict( zip(keys, values_out) )
	# dict_indeg = dict( zip(keys, values_in) )
	# # print values_in

	# c = float(sum(values_out))/num_nodes
	# print '[c = ] ' + str(c)
	# threshold = c * 2

	# #########################################
	# # prob > probability_upper: nothing written
	# # probability_upper > prob > probability_middle: write both
	# # probability_middle > prob > probability_lower: write new
	# # probability_lower > prob: write original
	# #########################################

	# probability_lower = 0.3333
	# probability_middle = 0.6666
	# probability_upper = 1.0

	# candidates = []

	# for i in range(num_nodes):
	# 	if dict_outdeg[i] >= threshold or dict_indeg[i] >= threshold:
	# 		candidates.append(i)

	# nodes_splitted = np.random.choice(candidates, TOTAL, replace=False)
	# nodes_splitted_dict = dict( zip(nodes_splitted, range(max_id+1, max_id + TOTAL+1)) )
	# # print '------------'
	# # print nodes_splitted_out
	# # print nodes_splitted_out_dict

	# # nodes_splitted_in = np.random.choice(candidates_in, TOTAL/2, replace=False)
	# # nodes_splitted_in_dict = dict( zip(nodes_splitted_in, range(max_id + TOTAL/2+1, max_id + TOTAL+1)) )

	# nodes_splitted_adj = sps.lil_matrix((TOTAL, num_nodes), dtype=int)
	# # nodes_splitted_in_adj = sps.lil_matrix((TOTAL/2, num_nodes), dtype=int)

	# counter = 0
	# for node in nodes_splitted:
	# 	if counter % 1000 == 0:
	# 		print '[Current node id]\t' + str(node) + '\t[new id] ' + str(counter)

	# 	orig_row = adj_matrix.getrow(node).toarray()[0].copy()
	# 	for k, v in enumerate(orig_row):
	# 		if v == 1:

	# 			temp = np.random.random()

	# 			if temp > probability_middle and temp <= probability_upper:
	# 				nodes_splitted_adj[counter, k] = 1
	# 			elif temp > probability_lower and temp <= probability_middle:
	# 				nodes_splitted_adj[counter, k] = 1
	# 				adj_matrix[node, k] = 0
	# 			else:
	# 				continue

	# 	counter += 1

	# output_matrix_3 = nodes_splitted_adj#sps.vstack([adj_matrix, ])

	# ################################################################################
	
	# counter = 0
	# adj_matrix = adj_matrix.T
	# nodes_splitted_adj_T = sps.lil_matrix((TOTAL, num_nodes), dtype=int)
	# for node in nodes_splitted:
	# 	if counter % 1000 == 0:
	# 		print '[Current node id]\t' + str(node) + '\t[new id] ' + str(counter)

	# 	orig_row = adj_matrix.getrow(node).toarray()[0].copy()
	# 	for k, v in enumerate(orig_row):
	# 		if v == 1:

	# 			temp = np.random.random()

	# 			if temp > probability_middle and temp <= probability_upper:
	# 				nodes_splitted_adj_T[counter, k] = 1
	# 			elif temp > probability_lower and temp <= probability_middle:
	# 				nodes_splitted_adj_T[counter, k] = 1
	# 				adj_matrix[node, k] = 0
	# 			else:
	# 				continue

	# 	counter += 1

	# adj_matrix = adj_matrix.T
	# output_matrix_1 = adj_matrix
	# output_matrix_2 = nodes_splitted_adj_T.T
	# output_matrix_4 = sps.lil_matrix((TOTAL, TOTAL), dtype=int)

	# for node_src in nodes_splitted:

	# 	for node_dst in nodes_splitted:

	# 		if adj_matrix[node_src, node_dst] == 1:

	# 			temp = np.random.random()

	# 			if temp > probability_middle and temp <= probability_upper:
	# 				output_matrix_4[nodes_splitted_dict[node_src] - max_id - 1, nodes_splitted_dict[node_dst] - max_id - 1] = 1
	# 			elif temp > probability_lower and temp <= probability_middle:
	# 				output_matrix_4[nodes_splitted_dict[node_src] - max_id - 1, nodes_splitted_dict[node_dst] - max_id - 1] = 1
	# 				adj_matrix[node_src, node_dst] = 0
	# 			else:
	# 				continue


	# adj_matrix_upper = sps.hstack([output_matrix_1, output_matrix_2])
	# adj_matrix_lower = sps.hstack([output_matrix_3, output_matrix_4])

	# adj_matrix = sps.vstack([adj_matrix_upper, adj_matrix_lower])





	# fOut = open(input_file_path.replace('.tsv', '_splitted.tsv'), 'w')


	# for idx in range(len(adj_matrix.row)):
	# 	src = adj_matrix.row[idx]
	# 	dst = adj_matrix.col[idx]
	# 	wei = int(adj_matrix.data[idx])
	# 	new_line = str(src) + delimiter + str(dst) + delimiter + str(wei) + '\n'
	# 	fOut.write(new_line)


	# fOut.close()

	# print '[max id new]' + str( max_id + TOTAL+1 )
	# print float(TOTAL)/float(num_nodes)

	# ############################################
	# # format: <orig_id> <new_id>
	# ############################################

	# fOut_s = open(input_file_path.replace('.tsv', '_splitted_new_nodes_mapping.tsv'), 'w')
	# for ele in nodes_splitted_dict:
	# 	fOut_s.write(str(ele) + delimiter + str(nodes_splitted_dict[ele]) + '\n')
	# fOut_s.close()


	# ############################################
	# # format: id-cat
	# ############################################
	# fOut_cat = open(input_file_path.replace('.tsv', '_splitted_cat.tsv'), 'w')
	
	# fIn = open(input_gt_path, 'r')
	# lines = fIn.readlines()
	# for line in lines:
	# 	fOut_cat.write(line)

	# fIn.close()

	# for node in nodes_splitted_dict:
	# 	new_line = str(ID_CAT_DICT[node]) + delimiter + str(nodes_splitted_dict[node]) + '\n'
	# 	fOut_cat.write(new_line)

	# fOut_cat.close()







