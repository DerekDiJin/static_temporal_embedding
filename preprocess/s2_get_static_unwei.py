import sys
import datetime
from pathlib import Path
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
import pickle
import itertools

import matplotlib
import matplotlib.pyplot as plt

import scipy.sparse as sps
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds, eigs
import sparsesvd
import scipy.spatial.distance as distance

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
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, DictionaryLearning
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from sklearn.feature_extraction import FeatureHasher
import collections
from collections import defaultdict



def bit_to_int(binary_list):
	# print binary_list
	return int(''.join([str(ele) for ele in binary_list]), 2)

def list_to_ints(l):
	'''
	Converts a list of values to str of ints
	'''
	# return ''.join(str(int(ele)) for ele in l)
	return [int(ele) for ele in l]

def combine_list(dict_in):
	result = []

	for ele in dict_in:
		result += dict_in[ele]
	return result



def emb_svd(feature_matrix, k):
	temp = scipy.sparse.csc_matrix(feature_matrix)
	U,s,V = sparsesvd.sparsesvd(temp, k)
	# S = sps.diag(s,dtype=float)
	#####################

	# U,s,V = svds(feature_matrix, k)
	S = np.diag(s)
	
	# print '----'
	# print U
	# print S
	emb = np.dot(U.T, (S ** 0.5))
	# g_sum = np.dot((S**0.5), V)

	return emb


def get_init_features(graph, base_features, nodes_to_explore):

	init_feature_matrix = np.zeros((len(nodes_to_explore), len(base_features)))
	adj = graph.adj_matrix

	if "outdegree" in base_features:
		init_feature_matrix[:,0] = (adj.sum(axis=0).transpose() +  adj.sum(axis=1)).ravel()

	if "indegree" in base_features:
		init_feature_matrix[:,1] = adj.sum(axis=0).transpose().ravel()

	if "degree" in base_features:
		init_feature_matrix[:,2] = adj.sum(axis=1).ravel()

	return init_feature_matrix






def create_reachability_weak(sub_dir, mod_str):
	for root, _, files in os.walk(sub_dir):
		print root, files

		total_counter = 0

		unique_file_set = set([])
		for file in files:

			file_number = file.split('.')[0].strip('_p_s')

			if file_number.isdigit():
				unique_file_set.add(int(file_number))

		total_counter = len(unique_file_set)

		print '====='
		print total_counter

		for idx in range( total_counter ):

			cur_file = str(idx) + '.tsv'
			relative_file_path = os.path.join(root, cur_file)
			abs_input_path = os.path.abspath(relative_file_path)

			cur_output = str(idx) + '_p.tsv'
			cur_snapshot_output = str(idx) + '_s.tsv'

			relative_file_path = os.path.join(root, cur_output)
			relative_snapshot_file_path = os.path.join(root, cur_snapshot_output)

			abs_output_path = os.path.abspath(relative_file_path)
			abs_snapshot_output_path = os.path.abspath(relative_snapshot_file_path)

			get_trg_t(abs_input_path, abs_snapshot_output_path, abs_output_path)

			# cur_output = str(idx) + '_s.tsv'
			# relative_file_path = os.path.join(root, cur_output)
			# abs_output_path = os.path.abspath(relative_file_path)
			# get_snapshot(abs_input_path, abs_output_path)

	return





#######################################################
# TODO: make sure edges in the temporal edgelist are unique
#######################################################
def parse_weighted_temporal(input_file_path, delimiter):

	check_eq = True
	num_nodes = 0
	num_edges = 0
	adj_matrix_global = None
	edge_time_dict = None
	time_edge_dict = None
	start_time = 0
	end_time = 0
	# node_roadmap = None


	raw = np.genfromtxt(input_file_path, dtype=int, delimiter=delimiter)
	print raw
	ROW, COL = raw.shape
	num_edges = ROW

	if COL == 3:
		print '[input_file does not contain timestamps. Processing as static graphs]'

		srcs = raw[:,0]
		dsts = raw[:,1]
		weis = raw[:,2]

		max_id = int(max(max(srcs), max(dsts)))
		num_nodes = max_id + 1
		print '[max_node_id] ' + str(max_id)
		print '[num_nodes] ' + str(num_nodes)

		if max(srcs) != max(dsts):
			srcs = np.append(srcs, max(max(srcs), max(dsts)))
			dsts = np.append(dsts, max(max(srcs), max(dsts)))
			weis = np.append(weis, 0)
			check_eq = False

		adj_matrix_global = sps.lil_matrix( sps.csc_matrix((weis, (srcs, dsts))))

	elif COL == 4:
		print '[input_file contains timestamps. Processing as dynamic graphs]'

		edge_time_dict = defaultdict(list)
		time_edge_dict = defaultdict(list)

		srcs = raw[:,0]
		dsts = raw[:,1]
		weis = raw[:,2]
		times = raw[:,3]

		start_time = min(times)
		end_time = max(times)

		max_id = int(max(max(srcs), max(dsts)))
		num_nodes = max_id + 1
		print '[max_node_id] ' + str(max_id)
		print '[num_nodes] ' + str(num_nodes)

		if max(srcs) != max(dsts):
			srcs = np.append(srcs, max(max(srcs), max(dsts)))
			dsts = np.append(dsts, max(max(srcs), max(dsts)))
			weis = np.append(weis, 0)
			check_eq = False

		adj_matrix_global = sps.lil_matrix( sps.csc_matrix((weis, (srcs, dsts))))

		fIn = open(input_file_path, 'r')
		lines = fIn.readlines()
		for line in lines:
			parts = line.strip('\r\n').split(delimiter)
			src = int(parts[0])
			dst = int(parts[1])
			wei = float(parts[2])
			timestamp = int(parts[3])

			edge = (src, dst, wei)
			edge_time_dict[edge].append(timestamp)
			time_edge_dict[timestamp].append(edge)


			# edge_r = (dst, src, wei)
			# edge_time_dict[edge_r].append(timestamp)
			# time_edge_dict[timestamp].append(edge_r)
			# tup = (dst, wei, timestamp)
			# node_roadmap[src].append(tup)

		fIn.close()


	else:
		sys.exit('[input_file format error. Please make sure the input file with the format <src, dst, wei> or <src, dst, wei, timestamps>')

	return check_eq, num_nodes, num_edges, adj_matrix_global, edge_time_dict, time_edge_dict, start_time, end_time


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


def find_bin_idx(bins, value):
	for idx in range(len(bins) - 1):
		if (value >= bins[idx]) and (value <= bins[idx+1]):
			return idx

	print value
	sys.exit('[Error in find_bin_idx]')
	return


def write_file_from_list(fOut, input_list):
	for line in input_list:
		fOut.write(line)

	return




def graph_split_number(input_file_path, E, T):

	result = defaultdict(list)

	step = E/T
	delimiter = find_delimiter(input_file_path)
	prefix, ext = os.path.splitext(input_file_path)
	sub_dir = prefix + '_subN_' + str(T)

	if not os.path.exists(sub_dir):
		os.mkdir(sub_dir)

	time_edge_list = []

	fIn = open(input_file_path, 'r')
	for line in fIn.readlines():
		src, dst, wei, timestamp_str = line.split(delimiter)

		timestamp = int(timestamp_str)
		time_edge_list.append((src, dst, wei, timestamp))

	fIn.close()

	time_edge_list.sort(key=lambda x:x[3])


	for idx, edge in enumerate(time_edge_list):
		cur_key = idx / (step + 1)
		cur_value = edge[0] + delimiter + edge[1] + delimiter + edge[2] + delimiter + str(edge[3]) + '\n'

		result[cur_key].append(cur_value)


	for idx in result:
		print '[Current idx] ' + str(idx) + ' [edge number] ' + str(len(result[idx]))

		cur_output_file_path = os.path.join(sub_dir, str(idx)+ext)

		fOut = open(cur_output_file_path, 'w')
		write_file_from_list(fOut, result[idx])
		fOut.close()


	return sub_dir


def graph_split_time_scale(input_temp_file_path, E, time_scale):

	delimiter = find_delimiter(input_temp_file_path)
	prefix, ext = os.path.splitext(input_temp_file_path)
	sub_dir = prefix + '_subTS_' + time_scale

	if time_scale == 'month':
		time_format = "%Y/%m"
	elif time_scale == 'day':
		time_format = "%Y/%m/%d"
	else:
		sys.exit('[Time_scale not supported.]')

	if not os.path.exists(sub_dir):
		os.mkdir(sub_dir)

	ET_sorted = []
	max_id = 0

	DICT_DAYS = defaultdict(int)
	DICT_EDGE_DAYS = defaultdict(list)

	fIn = open(input_temp_file_path, 'r')
	lines = fIn.readlines()
	fIn.close()

	for line in lines:
		src_str, dst_str, wei_str, temp_str = line.split('\t')
		src = int(src_str)
		dst = int(dst_str)
		wei = int(wei_str)
		temp = int(temp_str)

		if src == dst:
			continue
			
		temp_edge = (src, dst, wei, temp)

		ET_sorted.append(temp_edge)
		max_id = max_id if max_id > src else src
		max_id = max_id if max_id > dst else dst
		
		
		dt_object = datetime.datetime.fromtimestamp(temp)
		
		key = dt_object.strftime(time_format) #("%m/%d/%Y")
		DICT_DAYS[key] += 1
		DICT_EDGE_DAYS[key].append(line)
		
	print('[Total days] ' + str(len(DICT_DAYS)))

	ET_sorted.sort(key=lambda k:k[3])   

	# time_init = ET_sorted[0][3]
	# time_init_object = datetime.datetime.fromtimestamp(time_init)
		

	###################################################

	count_list = []

	for idx, ele in enumerate( sorted([datetime.datetime.strptime(key, time_format) for key in DICT_DAYS.keys()]) ):
		
		key = ele.strftime(time_format)
	#     print(key + '-' + str(DICT_DAYS[key]))
		count_list.append(DICT_DAYS[key])

		print '[Current idx] ' + str(idx) + ' [month] ' + key + ' [edge number] ' + str(len(DICT_EDGE_DAYS[key]))

		cur_output_file_path = os.path.join(sub_dir, str(idx)+ext)

		fOut = open(cur_output_file_path, 'w')
		write_file_from_list(fOut, DICT_EDGE_DAYS[key])
		fOut.close()

	return sub_dir






def graph_split_time(input_file_path, start_time, end_time, T):

	# TODO: change to batch format to support larger input graphs.
	result = defaultdict(list)

	delimiter = find_delimiter(input_file_path)
	prefix, ext = os.path.splitext(input_file_path)
	sub_dir = prefix + '_subT_' + str(T)

	if not os.path.exists(sub_dir):
		os.mkdir(sub_dir)

	bins = range(start_time, end_time+1, (end_time-start_time)/T)
	bins[-1] = end_time
	# counts, bins = np.histogram(range(start_time, end_time + 1), bins = T)
	print '[Input file path] ' + input_file_path
	print '[Splitted bins] ' + str(bins)

	fIn = open(input_file_path, 'r')
	for line in fIn.readlines():

		src, dst, wei, timestamp_str = line.split(delimiter)

		timestamp = int(timestamp_str)
		bin_idx = find_bin_idx(bins, timestamp)
		result[bin_idx].append(line)

	fIn.close()

	for idx in result:
		print '[Current idx] ' + str(idx) + ' [temporal range] ' + str(bins[idx]) + ' - ' + str(bins[idx+1])

		cur_output_file_path = os.path.join(sub_dir, str(idx)+ext)

		fOut = open(cur_output_file_path, 'w')
		write_file_from_list(fOut, result[idx])
		fOut.close()


	return sub_dir


def find_delimiter(input_file_path):
	delimiter = " "
	if ".csv" in input_file_path:
		delimiter = ","
	elif ".tsv" in input_file_path:
		delimiter = "\t"
	else:
		sys.exit('Format not supported.')

	return delimiter



def write_embedding(embedding_dict, output_file_path):

	first_key = next(iter(embedding_dict))
	first_ele = embedding_dict[first_key]

	N = len(embedding_dict)
	K = len(first_ele)

	fOut = open(output_file_path, 'w')
	fOut.write(str(N) + ' ' + str(K) + '\n')

	for i in range(N):
		cur_line = ' '.join([str(ii) for ii in embedding_dict[i]])
		fOut.write(str(i) + ' ' + cur_line + '\n')

	fOut.close()

	return




if __name__ == '__main__':
	
	if len(sys.argv) != 5:
		sys.exit('usage: stats_edges.py <snapshots_init,snapshots_end> <input_graph_file_path> <trg/snap> <T/N>')


	[init, end] = [int(ele) for ele in sys.argv[1].split(',')]
	input_graph_file_path = sys.argv[2]


	delimiter = find_delimiter(input_graph_file_path)

	''' adj_matrix: lil format adj matrix
		edge_time_dict: (src, dst, wei) - time_1, time_2, ...
	'''
	check_eq, num_nodes, num_edges, adj_matrix, edge_time_dict, time_edge_dict, start_time, end_time = parse_weighted_temporal(input_graph_file_path, delimiter)

	############################################################################################################
	# Setup
	############################################################################################################

	K = 128

	# TODO: add a list of supportive embedding methods here.
	# embed_method = 'node2vec'
	# embed_method = 'line'
	# embed_method = 'node2bits'
	# embed_method = 'role2vec'
	# embed_method = 'struc2vec'
	# embed_method = 'multilens'
	# embed_method = 'g2g'
	# embed_method = 'sgd'
	runMethod = True
	create_trg = True

	split_mod = sys.argv[3]#'T' # 'N' # 'TS'

	time_scale = sys.argv[4]

	print split_mod

	############################################################################################################

	# CAT_DICT, ID_CAT_DICT = construct_cat(input_gt_path, delimiter)

	print start_time, end_time


	#############################################
	# returns the sub directory of the splitted input grpah file.
	#############################################
	
	delimiter = find_delimiter(input_graph_file_path)
	prefix, ext = os.path.splitext(input_graph_file_path)



	sub_dir = prefix + '_sub' + split_mod + '_' + time_scale


	print input_graph_file_path
	print split_mod
	print sub_dir

	output_base_file_path = input_graph_file_path.split('/')[-1].split('.')[-2] + '_' + split_mod + '_bs.tsv'
	output_temp_file_path = input_graph_file_path.split('/')[-1].split('.')[-2] + '_' + split_mod + '.tsv'

	fOut_base = open(output_base_file_path, 'w')
	fOut_temp = open(output_temp_file_path, 'w')

	EDGE_COUNT_DICT = defaultdict(int)

	for idx in range(init, end+1):

		cur_file = sub_dir + '/' + str(idx) + '.tsv'

		fIn = open(cur_file, 'r')
		lines = fIn.readlines()
		fIn.close()

		for line in lines:

			src_str, dst_str, wei_str, time_str = line.strip().split('\t')
			edge = (int(src_str), int(dst_str))

			EDGE_COUNT_DICT[edge] += 1

			fOut_temp.write(line)


	for edge in EDGE_COUNT_DICT:
		fOut_base.write( str(edge[0]) + '\t' + str(edge[1]) + '\t' + str(EDGE_COUNT_DICT[edge]) + '\n' )

	fOut_base.close()
	fOut_temp.close()






	



