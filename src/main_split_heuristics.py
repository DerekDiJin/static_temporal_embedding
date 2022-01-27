import sys
import datetime
from pathlib import Path
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
import pickle
import itertools
from util import *

# import matplotlib
# import matplotlib.pyplot as plt

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



def get_trg(input_file_path, abs_snapshot_output_path, output_file_path):

	print '[Currently processing] ' + input_file_path + ' [output file] ' + output_file_path

	fOut = open(abs_snapshot_output_path, 'w')

	edge_list_temporal = []
	fIn = open(input_file_path, 'r')
	for line in fIn.readlines():
		src_str, dst_str, wei_str, time_str = line.strip().split('\t')
		
		src = int(src_str)
		dst = int(dst_str)
		wei = int(wei_str)
		timestamp = int(time_str)

		snapshot_line = src_str + '\t' + dst_str + '\t' + wei_str + '\n'
		fOut.write(snapshot_line)

		edge_list_temporal.append((src, dst, wei, timestamp))


	edge_list_temporal.sort(key=lambda x:x[3], reverse=True)

	fOut.close()
	################################################################

	result = defaultdict(int)
	Er = defaultdict(set)

	for idx, (i, j, w, t) in enumerate(edge_list_temporal):
		
		if idx % 1000 == 0:
			print idx
		
		if j in Er:
			for (dst, wei, temporal_path) in Er[j]:

				t_path_u = (t,) + temporal_path

				result[(i, dst)] += 1
				Er[i].add((dst, wei, t_path_u))

		Er[i].add( (j, w, (t,)) )
		result[(i, j)] += 1

		# print len(Er)

	fOut = open(output_file_path, 'w')
	for edge in result:
		fOut.write(str(edge[0]) + '\t' + str(edge[1]) + '\t' + str(result[edge]) + '\n')

	fOut.close()

	return


def get_trg_u(input_file_path, abs_snapshot_output_path, output_file_path):

	print '[Currently processing] ' + input_file_path + ' [output file] ' + output_file_path

	fOut = open(abs_snapshot_output_path, 'w')

	edge_list_temporal = []
	fIn = open(input_file_path, 'r')
	for line in fIn.readlines():
		src_str, dst_str, wei_str, time_str = line.strip().split('\t')
		
		src = int(src_str)
		dst = int(dst_str)
		wei = int(wei_str)
		timestamp = int(time_str)

		snapshot_line = src_str + '\t' + dst_str + '\t' + wei_str + '\n'
		fOut.write(snapshot_line)

		edge_list_temporal.append((src, dst, wei, timestamp))


	edge_list_temporal.sort(key=lambda x:x[3], reverse=True)

	fOut.close()
	################################################################

	result = set([])
	Er = defaultdict(set)

	for idx, (i, j, w, t) in enumerate(edge_list_temporal):
		
		if idx % 1000 == 0:
			print idx
		
		if j in Er:
			for (dst, wei, timestamp) in Er[j]:
				result.add((i, dst))
				Er[i].add((dst, wei, timestamp))

		Er[i].add((j, w, t))
		result.add((i, j))

		# print len(Er)

	fOut = open(output_file_path, 'w')
	for edge in result:
		fOut.write(str(edge[0]) + '\t' + str(edge[1]) + '\t1\n')

	fOut.close()

	return

def get_trg_t(input_file_path, abs_snapshot_output_path, output_file_path, start_time, end_time):

	print '[Currently processing] ' + input_file_path + ' [output file] ' + output_file_path

	duration = (end_time - start_time) * 1.0

	fOut = open(abs_snapshot_output_path, 'w')

	edge_list_temporal = []
	fIn = open(input_file_path, 'r')
	for line in fIn.readlines():
		src_str, dst_str, wei_str, time_str = line.strip().split('\t')
		
		src = int(src_str)
		dst = int(dst_str)
		wei = int(wei_str)
		timestamp = int(time_str)

		snapshot_line = src_str + '\t' + dst_str + '\t' + wei_str + '\n'
		fOut.write(snapshot_line)

		edge_list_temporal.append((src, dst, wei, timestamp))


	edge_list_temporal.sort(key=lambda x:x[3], reverse=True)

	fOut.close()
	################################################################

	result = defaultdict(int)
	Er = defaultdict(set)

	for idx, (i, j, w, t) in enumerate(edge_list_temporal):
		
		if idx % 1000 == 0:
			print idx
		
		if j in Er:
			for (dst, wei, timestamp) in Er[j]:
				result[(i, dst)] += np.exp( (t-timestamp) * 1.0 / duration ) # 1s
				Er[i].add((dst, wei, timestamp)) # t

		Er[i].add((j, w, t))
		result[(i, j)] += 1.0

		# print len(Er)

	fOut = open(output_file_path, 'w')
	for edge in result:
		fOut.write(str(edge[0]) + '\t' + str(edge[1]) + '\t' + str(round(result[edge], 6)) + '\n')

	fOut.close()

	return




def create_reachability_weak(sub_dir, mod_str, start_time, end_time):
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

		for idx in unique_file_set:

			cur_file = str(idx) + '.tsv'
			relative_file_path = os.path.join(root, cur_file)
			abs_input_path = os.path.abspath(relative_file_path)

			cur_output = str(idx) + '_p.tsv'
			cur_snapshot_output = str(idx) + '_s.tsv'

			relative_file_path = os.path.join(root, cur_output)
			relative_snapshot_file_path = os.path.join(root, cur_snapshot_output)

			abs_output_path = os.path.abspath(relative_file_path)
			abs_snapshot_output_path = os.path.abspath(relative_snapshot_file_path)

			# get_trg_t(abs_input_path, abs_snapshot_output_path, abs_output_path, start_time, end_time)
			get_trg_u(abs_input_path, abs_snapshot_output_path, abs_output_path)


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


def node_embed(sub_dir, eval_path, embed_method, K, num_nodes, runMethod, mod_str):

	#eval_path: "root/eval"#
	eval_path_cmd = eval_path.replace(' ', '\ ') + '/'
	output_path_prefix_cmd = os.path.join(eval_path.replace(' ', '\ '), embed_method, 'emb') + '/'


	for root, _, files in os.walk(sub_dir):
		print root, files

		total_counter = 0

		all_candidates = set([])

		for file in files:
			if mod_str in file:
				all_candidates.add(file)


		for candidate in all_candidates: #range( total_counter ):

			cur_file = candidate
			# cur_file = str(idx) + '_p.tsv'
			relative_file_path = os.path.join(root, cur_file)
			abs_input_path = os.path.abspath(relative_file_path).replace(' ', '\ ')	
			abs_cat_file_path = '/'.join(abs_input_path.split('/')[:-1])[:-4] + '_cat.tsv'
			# print '!!!', abs_cat_file_path

			output_file_path = output_path_prefix_cmd + abs_input_path.split('/')[-2] + '_' + abs_input_path.split('/')[-1].split('.')[-2] + '_emb.tsv'
			relative_file_path = os.path.join(root, output_file_path)
			abs_output_path = os.path.abspath(relative_file_path)

			print '========================'

			if embed_method == 'node2vec':
				cmd = 'python ' + eval_path_cmd + embed_method + '/src/main.py --input ' + abs_input_path + ' --output ' + abs_output_path + \
					' --p 1 --q 4 --dimensions ' + str(K) + ' --iter 5 --directed --weighted'#--weighted
			
			elif embed_method == 'line':
				cmd = eval_path_cmd + embed_method + '/line -train ' + '../' + '/'.join(abs_input_path.split('/')[-3:]) + ' -output ' + '../' + '/'.join(abs_output_path.split('/')[-4:]) + \
					' -size ' + str(K) + ' -order 2 -negative 5 -samples 10 -rho 0.025 -threads 1'

			elif embed_method == 'role2vec':
				cmd = 'python ' + eval_path_cmd + embed_method + '/src/main.py --graph-input ' + abs_input_path + ' --output ' + abs_output_path + \
					' --dimensions ' + str(K) + ' --features degree'

			elif embed_method == 'struc2vec':
				cmd = 'python ' + eval_path_cmd + embed_method + '/src/main.py --input ' + abs_input_path + ' --output ' + abs_output_path + \
					' --window-size 5 --dimensions ' + str(K) + ' --OPT1 True --OPT2 True --OPT3 True --until-layer 6'

			elif embed_method == 'node2bits':
				cmd = 'python ' + eval_path_cmd + embed_method + '/src/main.py ' + abs_input_path + ' ' + abs_cat_file_path + ' ' + abs_output_path 

			elif embed_method == 'multilens':
				cmd = 'python ' + eval_path_cmd + embed_method + '/src/main.py ' + abs_input_path + ' ' + abs_cat_file_path + ' ' + abs_output_path + \
					' ' + str(K)

			elif embed_method == 'g2g':
				cmd = 'python ' + eval_path_cmd + embed_method + '/run_g2g.py ' + abs_input_path + ' ' + abs_cat_file_path + ' ' + abs_output_path + \
					' ' + str(K)

			elif embed_method == 'sgd':
				cmd = 'python ' + eval_path_cmd + embed_method + '/run_sgd.py ' + abs_input_path + ' ' + abs_output_path + ' ' + str(num_nodes) + \
					' ' + str(K)

			elif embed_method == 'graphwave':
				cmd = 'python ' + eval_path_cmd + embed_method + '/run_graphwave.py ' + abs_input_path + ' ' + abs_output_path

			else:
				sys.exit('<Unsupported embedding method>')

			print cmd

			if runMethod:
				os.system(cmd)

	


	return output_path_prefix_cmd






def svd_emb(PSs, Ks):

	for idx in range(len(Ks)):
		Ki = Ks[idx]
		PSi = PSs[idx]
		emb_temp = feature_2_embedding(PSi, Ki)

		if idx == 0:
			emb = emb_temp
		else:
			emb = np.concatenate((emb, emb_temp), axis=1)

	return emb


def feature_2_embedding(feature_matrix = None, k = 17):

	####### TODO #######
	temp = sps.csc_matrix(feature_matrix)
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
	for tup in input_list:
		line = '\t'.join([str(e) for e in tup]) + '\n'
		fOut.write(line)

	return

def write_file_from_edge_list(fOut, input_edge_list):
	for src, dst, wei, time in input_edge_list:
		new_line = str(src) + '\t' + str(dst) + '\t' + str(wei) + '\t' + str(time) + '\n'
		fOut.write(new_line)

	return

def load_embeddings(filename):
	fIn = open(filename, 'r')
	node_num, size = [int(x) for x in fIn.readline().strip().split()]
	result = {}
	while 1:
		l = fIn.readline()
		if l == '':
			break
		vec = l.strip().split(' ')
		assert len(vec) == size+1
		result[int(float(vec[0]))] = [float(x) for x in vec[1:]]
	fIn.close()
	assert len(result) == node_num
	return result


def combine_embedding(sub_dir, input_emb_dir, total_num, embed_method, N, K, mod_emb_str):

	print 'Start combining embeddings.'

	combine_mod = 'sum'#'concatenatation' #'sum'#'concatenatation'

	result = defaultdict(list)

	prefix = sub_dir.split('/')[-1]
	# print 'LLLLLLL'
	# print input_file_name

	files_all = os.listdir(input_emb_dir.replace('\ ', ' '))
	# print files_all

	if combine_mod == 'concatenatation':

		

		for idx in range(total_num):

			cur_file = input_emb_dir.replace('\ ', ' ') + prefix + '_' + str(idx) + mod_emb_str

			cur_embedding_dict = load_embeddings(cur_file)

			for i in range(N):
				if i % 50000 == 0:
					print str(i) + ' / ' + str(N)
				if i in cur_embedding_dict:
					result[i] += cur_embedding_dict[i]
				else:
					result[i] += [0.0] * K

	elif combine_mod == 'sum':

		result_mx = np.zeros((N, K), dtype=float)

		theta = 1.

		for idx in range(total_num):

			cur_file = input_emb_dir.replace('\ ', ' ') + prefix + '_' + str(idx) + mod_emb_str
			print '[Combine file] ' + cur_file

			cur_embedding_dict = load_embeddings(cur_file)

			for nid in cur_embedding_dict:
				# emb = np.array([0.0] * K)
				# emb[:len(cur_embedding_dict[nid])] = np.array(cur_embedding_dict[nid]) * theta
				emb = np.array(cur_embedding_dict[nid]) * theta
				emb_last = result_mx[nid,:] * (1-theta)

				result_mx[nid,:] = emb + emb_last


		for i in range(N):
			result[i] = result_mx[i,:].tolist()





	else:
		sys.exit('[Combine mode Unsupported.]')




	return result


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


def get_time_scale_format(time_scale):
	if time_scale == 'month':
		time_format = "%Y/%m"
	elif time_scale == 'day':
		time_format = "%Y/%m/%d"
	elif time_scale == 'hour':
		time_format = "%Y/%m/%d/%H"
	else:
		sys.exit('[Time_scale not supported.]')

	return time_format



def graph_split_time_scale(input_temp_file_path, num_edges, train_init, train_num, pred_num, agg_factor, time_scale):

	delimiter = find_delimiter(input_temp_file_path)
	prefix, ext = os.path.splitext(input_temp_file_path)
	sub_dir = prefix + '_subTS_' + time_scale

	time_format = get_time_scale_format(time_scale)

	if not os.path.exists(sub_dir):
		os.mkdir(sub_dir)

	ET_sorted = []
	max_id = 0

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
		DICT_EDGE_DAYS[key].append(temp_edge)
		
	print('[Total days] ' + str(len(DICT_EDGE_DAYS)))

	ET_sorted.sort(key=lambda k:k[3])   

	# time_init = ET_sorted[0][3]
	# time_init_object = datetime.datetime.fromtimestamp(time_init)
		
	# DICT_EDGE_DAYS - <str> - [line, line, ...]

	###################################################
	# Re-order the temporal dict.
	###################################################
	IDX_EDGES_DICT = defaultdict(list)
	IDX_DATE_STR_DICT = defaultdict(str)

	for idx, ele in enumerate( sorted([datetime.datetime.strptime(key, time_format) for key in DICT_EDGE_DAYS.keys()]) ):
		
		key = ele.strftime(time_format)
	#     print(key + '-' + str(DICT_DAYS[key]))

		IDX_EDGES_DICT[idx] = DICT_EDGE_DAYS[key]
		IDX_DATE_STR_DICT[idx] = key


	###################################################
	# Aggregate according to the number.
	###################################################
	
	AGG_IDX_EDGES_DICT = defaultdict(list)

	cur_idx = 0
	cur_list = []

	for i in range(train_init, train_init + (train_num+pred_num) * agg_factor + 1):

		if int((i - train_init) / agg_factor) == cur_idx:
			cur_list += IDX_EDGES_DICT[i]

		else:
			cur_list.sort(key=lambda k:k[3])
			AGG_IDX_EDGES_DICT[cur_idx] = cur_list

			cur_idx += 1
			cur_list = IDX_EDGES_DICT[i]

	# print IDX_EDGES_DICT[train_init]
	# print '===='
	# print IDX_EDGES_DICT[train_init+1]
	# print '===='
	# print IDX_EDGES_DICT[train_init+2]
	# print '===='
	# print IDX_EDGES_DICT[train_init+3]
	# print '===='
	# print AGG_IDX_EDGES_DICT

	# sys.exit()


	###################################################
	# Write snapshots.
	###################################################
	# print AGG_IDX_EDGES_DICT

	write_snapshots(AGG_IDX_EDGES_DICT, range(train_num+pred_num), sub_dir, ext)
	# write_snapshots(pred_init, pred_num, AGG_IDX_EDGES_DICT, sub_dir, ext)

	return sub_dir, AGG_IDX_EDGES_DICT



def write_snapshots(AGG_IDX_EDGES_DICT, keys_list, sub_dir, ext):
	'''
	keys_list: idx of the snapshot
	'''
	for idx in keys_list: #or idx in range(pred_init, pred_end+1):

		print '[Current idx] ' + str(idx) + ' [edge number] ' + str(len(AGG_IDX_EDGES_DICT[idx]))

		cur_output_file_path = os.path.join(sub_dir, str(idx)+ext)

		fOut = open(cur_output_file_path, 'w')
		write_file_from_list(fOut, AGG_IDX_EDGES_DICT[idx])
		fOut.close()

	return sub_dir


def graph_split_number_scale_middle(input_temp_file_path, AGG_IDX_EDGES_DICT, train_init, train_num, pred_num, agg_factor, time_scale):

	delimiter = find_delimiter(input_temp_file_path)
	prefix, ext = os.path.splitext(input_temp_file_path)
	sub_dir = prefix + '_subNTS_' + time_scale

	time_format = get_time_scale_format(time_scale)

	if not os.path.exists(sub_dir):
		os.mkdir(sub_dir)


	edges_train = []
	for idx in range(train_num):
		edges_train += AGG_IDX_EDGES_DICT[idx]

	edges_amount = len(edges_train) / train_num

	print '[edges_amount] ' + str(edges_amount)

	AGG_IDX_EDGES_EQ_DICT = defaultdict(list)
	for idx in range(train_num):
		AGG_IDX_EDGES_EQ_DICT[idx] = edges_train[idx*edges_amount: (idx+1)*edges_amount]

	write_snapshots(AGG_IDX_EDGES_EQ_DICT, range(train_num), sub_dir, ext)
	write_snapshots(AGG_IDX_EDGES_DICT, range(train_num+pred_num)[(train_num):], sub_dir, ext)
	
	# for idx in range(train_num):
	# 	cur_output_file_path = os.path.join(sub_dir, str(idx)+ext)

	# 	cur_edge_list = edges_train[idx*edges_amount : (idx+1)*edges_amount]

	# 	fOut = open(cur_output_file_path, 'w')
	# 	write_file_from_edge_list(fOut, cur_edge_list)
	# 	fOut.close()



	return sub_dir


def tuple_2_line(tup):
	return '\t'.join([str(e) for e in tup]) + '\n'

def graph_split_number_scale(input_temp_file_path, AGG_IDX_EDGES_DICT, train_init, train_num, pred_num, agg_factor, time_scale):

	delimiter = find_delimiter(input_temp_file_path)
	prefix, ext = os.path.splitext(input_temp_file_path)
	sub_dir = prefix + '_subNS_' + time_scale

	time_format = get_time_scale_format(time_scale)

	if not os.path.exists(sub_dir):
		os.mkdir(sub_dir)

	ET_sorted = []
	max_id = 0

	DICT_LINE_DAYS = defaultdict(list)
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
		DICT_LINE_DAYS[key].append(line)
		DICT_EDGE_DAYS[key].append(temp_edge)
		
	print('[Total days] ' + str(len(DICT_LINE_DAYS)))

	ET_sorted.sort(key=lambda k:k[3])   

	# time_init = ET_sorted[0][3]
	# time_init_object = datetime.datetime.fromtimestamp(time_init)
		

	###################################################

	test_num = len(AGG_IDX_EDGES_DICT[train_num + pred_num - 1])
	marker = AGG_IDX_EDGES_DICT[train_num + pred_num - 1][0]
	marker_idx = ET_sorted.index(marker)
	# print marker_idx

	AGG_IDX_EDGES_EQ_DICT = defaultdict(list)
	for idx in range(train_num):
		AGG_IDX_EDGES_EQ_DICT[train_num - 1 - idx] = ET_sorted[marker_idx - test_num*(idx+1) : marker_idx - test_num*idx]

	write_snapshots(AGG_IDX_EDGES_EQ_DICT, range(train_num), sub_dir, ext)
	write_snapshots(AGG_IDX_EDGES_DICT, range(train_num+pred_num)[(train_num):], sub_dir, ext)

	return sub_dir

	# DICT_IDX_DAYS = {}

	# for idx, ele in enumerate( sorted([datetime.datetime.strptime(key, time_format) for key in DICT_LINE_DAYS.keys()]) ):
		
	# 	key = ele.strftime(time_format)
	# 	DICT_IDX_DAYS[idx] = key
	# #     print(key + '-' + str(DICT_DAYS[key]))
	# 	print '[Current idx] ' + str(idx) + ' [' + time_scale + '] ' + key + ' [edge number] ' + str(len(DICT_EDGE_DAYS[key]))



	# pred_edges = DICT_EDGE_DAYS[DICT_IDX_DAYS[pred_init]]
	
	# pred_edges.sort(key=lambda k:k[3])
	# marker = pred_edges[0]
	# marker_idx = ET_sorted.index(marker)
	# print marker_idx


	# pred_E_count = 0
	# for idx in range(pred_init, pred_end+1):
	# 	pred_E_count += len(DICT_EDGE_DAYS[DICT_IDX_DAYS[idx]])

	# cur_output_file_path = os.path.join(sub_dir, str(pred_init)+ext)
	# cur_edge_list = ET_sorted[marker_idx : marker_idx + pred_E_count]

	# fOut = open(cur_output_file_path, 'w')
	# write_file_from_edge_list(fOut, cur_edge_list)
	# fOut.close()

	# print pred_E_count

	# train_duration = train_end - train_init + 1
	# for idx in range(train_duration):
	# 	cur_output_file_path = os.path.join(sub_dir, str(pred_init - idx - 1)+ext)

	# 	cur_edge_list = ET_sorted[marker_idx - pred_E_count*(idx+1) : marker_idx - pred_E_count*idx]

	# 	fOut = open(cur_output_file_path, 'w')
	# 	write_file_from_edge_list(fOut, cur_edge_list)
	# 	fOut.close()



	# return sub_dir






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



def write_embedding(embedding_dict, output_file_path, N):

	fOut = open(output_file_path, 'w')
	fOut.write(str(N) + ' ' + str(len( embedding_dict[next(iter(embedding_dict))] )) + '\n')

	for i in range(N):
		cur_line = ' '.join([str(ii) for ii in embedding_dict[i]])
		fOut.write(str(i) + ' ' + cur_line + '\n')

	fOut.close()

	return




if __name__ == '__main__':
	
	if len(sys.argv) != 10:
		sys.exit('usage: stats_edges.py <input_snapshot_count> <trg/snap> <T/N> <embed_method> <train_init> <train_end> <test_init> <test_end> <iter>')


	# assume the graph is undirected, unweighted
	weighted = True
	directed = True

	# input_file_path = sys.argv[1]
	cur_file_path = str(Path().resolve().parent)
	eval_path = os.path.join(cur_file_path, 'eval')
	print eval_path

	# input_file_path = cur_file_path + '/toy_graphs_lp_time/ia-contacts_hypertext2009_p.tsv'
	# input_gt_path = cur_file_path + '/toy_graphs_lp_time/ia-contacts_hypertext2009_p_cat.tsv'

	# input_file_path = cur_file_path + '/toy_graphs_lp_time/ia-hospital-ward-proximity_p.tsv'
	# input_gt_path = cur_file_path + '/toy_graphs_lp_time/ia-hospital-ward-proximity_cat.tsv'

	# input_file_path = cur_file_path + '/toy_graphs_lp_time/fb-forum_p.tsv' # 7days 2
	# input_gt_path = cur_file_path + '/toy_graphs_lp_time/fb-forum_p_cat.tsv' # 7days
	# input_file_path = cur_file_path + '/toy_graphs_lp_time/reality-call_p.tsv' # 7days
	# input_gt_path = cur_file_path + '/toy_graphs_lp_time/reality-call_p_cat.tsv' # 7days
	# input_file_path = cur_file_path + '/toy_graphs_lp_time/ia-contacts_dublin_p.tsv' # days
	# input_gt_path = cur_file_path + '/toy_graphs_lp_time/ia-contacts_dublin_p_cat.tsv' # days

	# input_file_path = cur_file_path + '/small_graphs_lp_time_up/soc-sign-bitcoinalpha_wei_temp.tsv'
	# input_gt_path = cur_file_path + '/small_graphs_lp_time_up/soc-sign-bitcoinalpha_cat.tsv'
	input_file_path = cur_file_path + '/small_graphs_lp_time/soc-sign-bitcoinalpha_wei_temp.tsv'
	input_gt_path = cur_file_path + '/small_graphs_lp_time/soc-sign-bitcoinalpha_cat.tsv'
	# input_file_path = cur_file_path + '/small_graphs_lp_time/sx-mathoverflow_wei_temp.tsv'
	# input_gt_path = cur_file_path + '/small_graphs_lp_time/sx-mathoverflow_cat.tsv'
	# input_file_path = cur_file_path + '/small_graphs_lp_time/wiki-elec_p.tsv'
	# input_gt_path = cur_file_path + '/small_graphs_lp_time/wiki-elec_p_cat.tsv'
	# input_file_path = cur_file_path + '/small_graphs_lp_time/ia-enron-employees_p.tsv'
	# input_gt_path = cur_file_path + '/small_graphs_lp_time/ia-enron-employees_p_cat.tsv'
	# input_file_path = cur_file_path + '/large_graphs_lp_time/digg-friends_p.tsv'
	# input_gt_path = cur_file_path + '/large_graphs_lp_time/digg-friends_p_cat.tsv'
	# input_file_path = cur_file_path + '/small_graphs_lp_time/wikipedia_p.tsv'
	# input_gt_path = cur_file_path + '/small_graphs_lp_time/wikipedia_p_cat.tsv'

	# input_file_path = cur_file_path + '/small_graphs_lp/sx-mathoverflow_wei_temp_train.tsv'
	# input_gt_path = cur_file_path + '/small_graphs_lp/sx-mathoverflow_cat.tsv'
	# input_file_path = cur_file_path + '/adobe_graphs/1166-2017-12_p.tsv'
	# input_gt_path = cur_file_path + '/adobe_graphs/1166-2017-12_cat.tsv'
	# input_file_path = cur_file_path + '/small_graphs/test.tsv'
	# input_gt_path = cur_file_path + '/small_graphs/test_cat.tsv'

	# input_file_path = cur_file_path + '/small_graphs_lp_time/soc-sign-bitcoinalpha_wei_temp_train.tsv'
	# input_gt_path = cur_file_path + '/small_graphs_lp_time/soc-sign-bitcoinalpha_cat.tsv'
	# input_file_path = cur_file_path + '/small_graphs_lp_time/sx-mathoverflow_wei_temp_train.tsv'
	# input_gt_path = cur_file_path + '/small_graphs_lp_time/sx-mathoverflow_cat.tsv'
	# input_file_path = cur_file_path + '/large_graphs_lp_time/wiki-talk-temporal_p_train.tsv'
	# input_gt_path = cur_file_path + '/large_graphs_lp_time/wiki-talk-temporal_cat.tsv'



	delimiter = find_delimiter(input_file_path)

	''' adj_matrix: lil format adj matrix
		edge_time_dict: (src, dst, wei) - time_1, time_2, ...
	'''
	check_eq, num_nodes, num_edges, adj_matrix, edge_time_dict, time_edge_dict, start_time, end_time = parse_weighted_temporal(input_file_path, delimiter)

	############################################################################################################
	# Setup
	############################################################################################################

	walks_num = 10
	walk_length = 20
	nodes_to_explore = range(num_nodes)
	base_features = ['degree', 'indegree', 'outdegree']

	K = 128
	num_buckets = 4	#2
	bucket_max_value = 30


	# TODO: add a list of supportive embedding methods here.
	embed_method = sys.argv[3] #'node2vec'
	# embed_method = 'line'
	# embed_method = 'node2bits'
	# embed_method = 'role2vec'
	# embed_method = 'struc2vec'
	# embed_method = 'multilens'
	# embed_method = 'g2g'
	# embed_method = 'sgd'
	runMethod = True
	create_trg = True
	emb_combining = True

	mod = sys.argv[1]#'trg'#'snapshot' 'trg'
	split_mod = sys.argv[2]#'T' # 'N' # 'TS'

	time_scale = sys.argv[8] #'hour'#'month'

	train_init = int(sys.argv[4])
	
	agg_factor = int(sys.argv[5])

	train_num = int(sys.argv[6])
	pred_num = int(sys.argv[7])

	ITER = sys.argv[9]

	print mod
	print split_mod

	############################################################################################################

	if mod == 'snapshot':
		mod_str = '_s'
	elif mod == 'trg':
		mod_str = '_p'
	else:
		sys.exit('[Mod Error]')

	mod_emb_str = mod_str + '_emb.tsv'

	CAT_DICT, ID_CAT_DICT = construct_cat(input_gt_path, delimiter)

	print start_time, end_time

	if len(ID_CAT_DICT) != num_nodes:
		print 'Inconsistent number of nodes. Corrected.'
		num_nodes = len(ID_CAT_DICT)

	#############################################
	# returns the sub directory of the splitted input grpah file.
	#############################################
	
	if split_mod == 'TS':
		sub_dir, AGG_IDX_EDGES_DICT = graph_split_time_scale(input_file_path, num_edges, train_init, train_num, pred_num, agg_factor, time_scale)
	elif split_mod == 'NS':
		sub_dir, AGG_IDX_EDGES_DICT = graph_split_time_scale(input_file_path, num_edges, train_init, train_num, pred_num, agg_factor, time_scale)
		sub_dir = graph_split_number_scale(input_file_path, AGG_IDX_EDGES_DICT, train_init, train_num, pred_num, agg_factor, time_scale)
	elif split_mod == 'NTS':
		sub_dir, AGG_IDX_EDGES_DICT = graph_split_time_scale(input_file_path, num_edges, train_init, train_num, pred_num, agg_factor, time_scale)
		sub_dir = graph_split_number_scale_middle(input_file_path, AGG_IDX_EDGES_DICT, train_init, train_num, pred_num, agg_factor, time_scale)
	else:
		sys.exit(['Split Mod Error'])

	

	#############################################
	# from the splitted graph files to "<splitted>_p" files
	#############################################
	if create_trg:
		create_reachability_weak(sub_dir, mod_str, start_time, end_time)

	emb_dir = node_embed(sub_dir, eval_path, embed_method, K, num_nodes, runMethod, mod_str)


	if emb_combining:
		embedding_dict = combine_embedding(sub_dir, emb_dir, (train_num), embed_method, num_nodes, K, mod_emb_str)
		output_file_path = 'emb' + ITER + '/' + input_file_path.split('/')[-1].split('.')[-2] + '_' + embed_method +  '_' + split_mod + mod_emb_str
		write_embedding(embedding_dict, output_file_path, num_nodes)

	





