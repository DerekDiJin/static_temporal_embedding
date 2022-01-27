import sys
import datetime
from pathlib import Path
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
import pickle
import itertools

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

			cur_file = input_emb_dir.replace('\ ', ' ') + sub_dir.split('/')[1] + '_' + str(idx) + mod_emb_str

			cur_embedding_dict = load_embeddings(cur_file)

			for i in range(N):
				if i % 50000 == 0:
					print str(i) + ' / ' + str(N)
				if i in cur_embedding_dict:
					result[i] += cur_embedding_dict[i][:(K / total_num)]
				else:
					result[i] += [0.0] * (K / total_num)

	elif combine_mod == 'sum':

		result_mx = np.zeros((N, K), dtype=float)

		theta = 0.8

		for idx in range(total_num):

			# import pdb; pdb.set_trace()
			cur_file = input_emb_dir.replace('\ ', ' ') + sub_dir.split('/')[1] + '_' + str(idx) + mod_emb_str
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
		fOut.write('\t'.join(str(ele) for ele in line) + '\n')

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

def write_embedding(embedding_dict, output_file_path, N):

	fOut = open(output_file_path, 'w')
	fOut.write(str(N) + ' ' + str(len( embedding_dict[next(iter(embedding_dict))] )) + '\n')

	for i in range(N):
		cur_line = ' '.join([str(ii) for ii in embedding_dict[i]])
		fOut.write(str(i) + ' ' + cur_line + '\n')

	fOut.close()

	return


def reorder(input_dir, new_splits, embed_method, split_mod, mod_str):

	result_N = defaultdict(list)

	all_files = os.listdir(input_dir)


	timestamps_set = set([])

	num_all_files = len(all_files) / 3 - 1
	all_edges = []
	print(num_all_files)

	test_file = '{}/{}.tsv'.format(input_dir, num_all_files)


	for idx in range(num_all_files):
		cur_input_file = '{}/{}.tsv'.format(input_dir,idx)

		fIn = open(cur_input_file, 'r')
		for line in fIn.readlines():
			src, dst, wei, timestamp_str = line.split('\t')
			timestamp = int(timestamp_str)
			timestamps_set.add(timestamp)
			all_edges.append((src, dst, wei, timestamp))
		fIn.close()

	step = len(all_edges) / new_splits

	all_edges.sort(key=lambda x:x[3], reverse=True)

	print(all_edges[:20])

	for idx, edge in enumerate(all_edges):
		cur_key = idx / (step + 1)
		# cur_value = '\t'.join([str(ele) for ele in edge]) + '\n'
		result_N[new_splits - cur_key].append(edge)


	for ele in result_N:
		result_N[ele].sort(key=lambda x:x[3])

	sub_dir_N = './{}_N/'.format(new_splits)
	os.system('mkdir {}_N'.format(new_splits))

	for idx in result_N:
		print '[Current idx] ' + str(idx) + ' [edge number] ' + str(len(result_N[idx]))
		cur_output_file_path = os.path.join(sub_dir_N, str(idx-1)+'.tsv')

		fOut = open(cur_output_file_path, 'w')
		write_file_from_list(fOut, result_N[idx])
		fOut.close()

	os.system('cp {} {}'.format(test_file, sub_dir_N + str(new_splits) + '.tsv'))


	############################

	result_T = defaultdict(list)

	timestamp_min = min(timestamps_set)
	timestamp_max = max(timestamps_set)

	print(timestamp_min, timestamp_max)

	step = (timestamp_max - timestamp_min) / new_splits

	for idx, edge in enumerate(all_edges):

		cur_timestamp = edge[3]
		cur_key = (cur_timestamp - timestamp_min) / (step + 1)
		# cur_value = '\t'.join([str(ele) for ele in edge]) + '\n'
		result_T[new_splits - cur_key].append(edge)

	for ele in result_T:
		result_T[ele].sort(key=lambda x:x[3])

	sub_dir_T = './{}_T/'.format(new_splits)
	os.system('mkdir {}_T'.format(new_splits))

	for idx in result_T:
		print '[Current idx] ' + str(idx) + ' [edge number] ' + str(len(result_T[idx]))
		cur_output_file_path = os.path.join(sub_dir_T, str(idx-1)+'.tsv')

		fOut = open(cur_output_file_path, 'w')
		write_file_from_list(fOut, result_T[idx])
		fOut.close()

	os.system('cp {} {}'.format(test_file, sub_dir_T + str(new_splits) + '.tsv'))


	node_set = set([])
	for idx in range(num_all_files):
		cur_input_file = '{}/{}.tsv'.format(input_dir,idx)

		fIn = open(cur_input_file, 'r')
		for line in fIn.readlines():
			src, dst, wei, timestamp_str = line.split('\t')
			node_set.add(int(src))
			node_set.add(int(dst))
			
		fIn.close()

	num_nodes = max(node_set) + 1

	if split_mod == 'TS':
		sub_dir = sub_dir_T
	else:
		sub_dir = sub_dir_N

	create_reachability_weak(sub_dir, mod_str, timestamp_min, timestamp_max)


	cur_file_path = str(Path().resolve().parent)
	eval_path = os.path.join(cur_file_path, 'eval')
	print eval_path

	mod_emb_str = mod_str + '_emb.tsv'

	K = 128
	runMethod = True




	emb_dir = node_embed(sub_dir, eval_path, embed_method, K, num_nodes, runMethod, mod_str)
	print(num_all_files)
	embedding_dict = combine_embedding(sub_dir, emb_dir, (new_splits), embed_method, num_nodes, K, mod_emb_str)
	output_file_path = 'emb' + '1' + '/' + input_dir + '_' + embed_method +  '_' + split_mod + mod_emb_str
	write_embedding(embedding_dict, output_file_path, num_nodes)

	return






if __name__ == '__main__':
	
	if len(sys.argv) != 6:
		sys.exit('usage: reorder.py <input_dir> <new_splits> <embed_method> <split_mod> <mod_str>')

	input_dir = sys.argv[1]
	new_splits = int(sys.argv[2])
	embed_method = sys.argv[3]
	split_mod = sys.argv[4]
	mod_str = sys.argv[5]

	reorder(input_dir, new_splits, embed_method, split_mod, mod_str)

	






	



