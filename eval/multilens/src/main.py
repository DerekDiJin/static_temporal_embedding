import sys
import datetime
from pathlib import Path
import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys, random
from collections import deque
import pickle
from util import *

# import matplotlib
# import matplotlib.pyplot as plt

import scipy.sparse as sps
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds, eigs
import sparsesvd

from collections import defaultdict

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



#Get the combined degree/other feature sequence for a given node
def get_combined_feature_sequence(graph, rep_method, current_node, input_dense_matrix = None, feature_wid_ind = None):
	
	N, cur_P = input_dense_matrix.shape

	id_cat_dict = graph.id_cat_dict
	combined_feature_vector = []
	cur_neighbors = graph.neighbor_list[current_node][:] #list(graph.adj_matrix.getrow(current_node).nonzero()[1]) #list(graph.nx_graph.neighbors(current_node))
	# print len(cur_neighbors)
	# print cur_neighbors
	cur_neighbors.append(current_node)

	# if current_node == N-1 and graph.check_eq == False: 
	# 	cur_neighbors.remove(current_node)

	# print cur_neighborhood
	
	for cat in graph.cat_dict.keys():

		features = []
		for i in range(cur_P):
			features.append([0.0] * feature_wid_ind[i])

		for neighbor in cur_neighbors:

			if id_cat_dict[neighbor] != cat:
				continue			

			try:
				# print cur_P
				for i in range(cur_P):
					node_feature = input_dense_matrix[neighbor, i]

					if (rep_method.num_buckets is not None) and (node_feature != 0):
						bucket_index = max( int(math.log(node_feature, rep_method.num_buckets)), 0 )
					else:
						bucket_index = int(node_feature)

					bucket_index = max(bucket_index, 0)


					features[i][min(bucket_index, len(features[i]) - 1)] += 1
			except Exception as e:
				# print current_node
				# print neighbor
				# print i
				# print input_dense_matrix.shape
				# print node_feature
				# print rep_method.num_buckets
				# print bucket_index
				print "Exception:", e, node_feature, bucket_index
		cur_feature_vector = features[0]
		
		for feature_vector in features[1:]:
			cur_feature_vector += feature_vector

		combined_feature_vector += cur_feature_vector
	
	return combined_feature_vector

#Get structural features for nodes in a graph based on degree sequences of neighbors
#Input: adjacency matrix of graph
#Output: nxD feature matrix
def get_features(graph, rep_method, input_dense_matrix = None, verbose = False, nodes_to_embed = None):
	before_khop = time.time()


	feature_wid_sum, feature_wid_ind = get_feature_n_buckets(input_dense_matrix, num_buckets, rep_method.bucket_max_value)
	feature_matrix = np.zeros([graph.num_nodes, feature_wid_sum * len(graph.unique_cat)])

	before_degseqs = time.time()

	for n in nodes_to_embed:
		if n % 10000 == 0:
			print "[Generate combined feature vetor] node: " + str(n)
		combined_feature_sequence = get_combined_feature_sequence(graph, rep_method, n, input_dense_matrix = input_dense_matrix, feature_wid_ind = feature_wid_ind)
		feature_matrix[n,:] = combined_feature_sequence

	after_degseqs = time.time() 

	if verbose:
		print "got degree sequences in time: ", after_degseqs - before_degseqs

	return feature_matrix


def get_seq_features(graph, rep_method, input_dense_matrix = None, verbose = False, nodes_to_embed = None):
	
	if input_dense_matrix is None:
		sys.exit('get_seq_features: no input matrix.')

	if nodes_to_embed is None:
		nodes_to_embed = range(graph.num_nodes)
		num_nodes = graph.num_nodes
	else:
		num_nodes = len(nodes_to_embed)

	if verbose:
		print "Nodes to embed: ", nodes_to_embed
	
	feature_matrix = get_features(graph, rep_method, input_dense_matrix, verbose, nodes_to_embed)
	print "Feature matrix: ", feature_matrix

	if graph.directed:
		print "got outdegree degseqs, now getting indegree..."

		neighbor_list_r = construct_neighbor_list(graph.adj_matrix.transpose(), nodes_to_embed)

		indegree_graph = Graph(graph.adj_matrix.transpose(),  max_id = graph.max_id, num_nodes = graph.num_nodes, 
			weighted = graph.weighted, directed = graph.directed, base_features = graph.base_features, neighbor_list = neighbor_list_r,
			cat_dict = graph.cat_dict, id_cat_dict = graph.id_cat_dict, unique_cat = graph.unique_cat, check_eq = graph.check_eq)
		base_feature_matrix_in = get_features(indegree_graph, rep_method, input_dense_matrix, verbose, nodes_to_embed = nodes_to_embed)

		print "Indegree feature matrix: ", base_feature_matrix_in
		print base_feature_matrix_in.shape

		feature_matrix = np.hstack((feature_matrix, base_feature_matrix_in))

	if verbose:
		print "Dimensionality in explicit feature space: ", feature_matrix.shape

	# 	#run struc2vec on that file
	# 	#TODO configure struc2vec so that on request, it can compute a similarity matrix, write it to a file and then stop, not train a language model
	# 	#TODO make it write to output and stop, not write embeddings to output
	# 	os.system("python baseline_repr_learning/struc2vec-modified/src/main.py --input input.txt --output output.emb --workers 1")

	# 	#read in struc2vec similarity matrix 
	# 	struc2vec_sim_matrix = np.load("sim.npy")

	# 	#clean up
	# 	os.system("rm input.txt")
	# 	os.system("rm sim.npy")
	
	return feature_matrix

	# if rep_method.normalize:
	# 	print("normalizing...")
	# 	norms = np.linalg.norm(reprsn, axis = 1).reshape((reprsn.shape[0],1))
	# 	norms[norms == 0] = 1 #TODO figure out why getting zeros representation
	# 	reprsn = reprsn / norms



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
	return result, result.keys(), id_cat_dict

######################################################

def search_feature_layer(graph, rep_method, base_feature_matrix = None):

	n,p = base_feature_matrix.shape

	result = np.zeros([n, p*rep_method.use_total])

	for u in range(n):
		if u % 10000 == 0:
			print '[Current_node_id] ' + str(u)

		neighbors = graph.neighbor_list[u]#graph.adj_matrix.getrow(u).nonzero()[1] #list(graph.nx_graph.neighbors(u))

		# Option: adding sampling here for speed?

		for fid in range(p):

			mean_v = 0.0; sum_v = 0.0; var_v = 0.0; max_v = 0.0; min_v = 0.0; sum_sq_diff = 0.0; prod_v = 1.0; L1_v = 0.0; L2_v = 0.0

			for v in neighbors:

				L1_v += abs(base_feature_matrix[u][fid] - base_feature_matrix[v][fid])	# L1

				diff = base_feature_matrix[u][fid] - base_feature_matrix[v][fid]

				L2_v += diff*diff	# L2

				sum_sq_diff += base_feature_matrix[v][fid] * base_feature_matrix[v][fid]     # var

				sum_v += base_feature_matrix[v][fid]  # used in sum and mean
									  
				# if base_feature_matrix[v][fid] > 0:	# prod
				# 	prod_v *= base_feature_matrix[v][fid] 
				if max_v < base_feature_matrix[v][fid]:	# max
					max_v = base_feature_matrix[v][fid]
				if min_v > base_feature_matrix[v][fid]:      # min
					min_v = base_feature_matrix[v][fid]

			deg = len(neighbors)
			if deg == 0:
				mean_v = 0
				var_v = 0
			else:
				mean_v = sum_v / float(deg)
				var_v = abs((sum_sq_diff / float(deg)) - (mean_v * mean_v)) #- 2.0*mean_v/float(deg)*sum_v

			temp_vec = [0.0] * rep_method.use_total
			temp_vec[0] = mean_v
			temp_vec[1] = var_v
			temp_vec[2] = sum_v
			temp_vec[3] = max_v
			temp_vec[4] = min_v
			temp_vec[5] = L1_v
			temp_vec[6] = L2_v
			#############################
			# temp_vec[0] = max_v
			# temp_vec[1] = min_v
			# temp_vec[0] = sum_v
			# temp_vec[1] = mean_v
			# temp_vec[4] = var_v
			
			# temp_vec[5] = L1_v
			# temp_vec[6] = L2_v

			result[u, fid*rep_method.use_total:(fid+1)*rep_method.use_total] = temp_vec

	return result



def feature_layer_filter(feature_matrix):

	output = None

	if feature_matrix is not None:

		n, p = feature_matrix.shape
		zero_col_idx = np.where(~feature_matrix.any(axis=0))[0]
		output = np.delete(feature_matrix, zero_col_idx, axis=1)

	return output


def feature_layer_evaluation_embedding(graph, rep_method, feature_matrix = None, k = 17):

	####### TODO #######
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
	g_sum = np.dot((S**0.5), V)

	return emb, g_sum

def row_normalizer(feature_matrix_emb):
	# row_sums = feature_matrix_emb.sum(axis=1) + 1e-6
	col_sums = feature_matrix_emb.sum(axis=0) + 1e-6
	return feature_matrix_emb / col_sums#feature_matrix_emb.max()

def construct_neighbor_list(adj_matrix, nodes_to_embed):
	result = {}

	for i in nodes_to_embed:
		result[i] = list(adj_matrix.getrow(i).nonzero()[1])

	return result



def get_init_features(graph, base_features, nodes_to_embed):

	###################################
	# set operator: sum as default.
	###################################

	init_feature_matrix = np.zeros((len(nodes_to_embed), len(base_features)), dtype=float)
	adj = graph.adj_matrix

	if "row_col" in base_features:
		init_feature_matrix[:,base_features.index("row_col")] = (adj.sum(axis=0).transpose() +  adj.sum(axis=1)).ravel()

	if "col" in base_features:
		init_feature_matrix[:,base_features.index("col")] = adj.sum(axis=0).transpose().ravel()

	if "row" in base_features:
		init_feature_matrix[:,base_features.index("row")] = adj.sum(axis=1).ravel()

	return init_feature_matrix


def get_init_features_all(graph, rep_method, base_features, nodes_to_embed, neighbor_list, neighbor_list_r):

	operators_num = rep_method.use_total	#max, min, sum, mean, var, l1, l2

	init_feature_matrix = np.zeros((len(nodes_to_embed), operators_num*len(base_features)))
	adj = graph.adj_matrix
	adj_r = adj.T

	for feature in base_features:
		print 'Current feature: ' + feature

		cur_idx = base_features.index(feature) * operators_num

		for i in nodes_to_embed:
			if i % 10000 == 0:
				print i

			b_v = []
			if feature == 'row':

				neighbors = neighbor_list[i][:]
				neighbors.append(i)
				for neighbor in neighbors:
					b_v.append(adj[i, neighbor])
				
			elif feature == 'col':
				neighbors = neighbor_list_r[i][:]
				neighbors.append(i)
				for neighbor in neighbors:
					b_v.append(adj_r[i, neighbor])

			elif feature == 'row_col':
				cur_row = neighbor_list[i][:]
				for neighbor in cur_row:
					b_v.append(adj[i, neighbor])

				cur_col = neighbor_list_r[i][:]
				neighbors.append(i)
				for neighbor in cur_col:
					b_v.append(adj_r[i, neighbor])

			else:
				sys.exit('Base feature not supported.')



			# init_feature_matrix[i, 0+cur_idx] = np.max(b_v)
			# init_feature_matrix[i, 1+cur_idx] = np.min(b_v)
			init_feature_matrix[i, 0+cur_idx] = np.sum(b_v)
			init_feature_matrix[i, 1+cur_idx] = np.mean(b_v)
			# init_feature_matrix[i, 4+cur_idx] = np.var(b_v)

			cur_val = adj[i,i]

			# init_feature_matrix[i, 5+cur_idx] = np.sum( [abs(cur_val - ele) for ele in b_v] )
			# init_feature_matrix[i, 6+cur_idx] = np.sum( [(cur_val - ele)*(cur_val - ele) for ele in b_v] )


	print 'Initial_feature_all finished.'
	print init_feature_matrix

	return init_feature_matrix



def get_feature_n_buckets(feature_matrix, num_buckets, bucket_max_value):

	result_sum = 0
	result_ind = []
	N, cur_P = feature_matrix.shape

	if num_buckets is not None:
		for i in range(cur_P):
			temp = max(bucket_max_value, int(math.log(max(max(feature_matrix[:,i]), 1), num_buckets) + 1))
			n_buckets = temp
			# print max(feature_matrix[:,i])
			result_sum += n_buckets
			result_ind.append(n_buckets)
	else:
		for i in range(cur_P):
			temp = max(bucket_max_value, int( max(feature_matrix[:,i]) ) + 1)
			n_buckets = temp
			result_sum += n_buckets
			result_ind.append(n_buckets)

	return result_sum, result_ind


def get_Kis(init_feature_matrix_seq, K, L):
	
	result = []
	rank_init = np.linalg.matrix_rank(init_feature_matrix_seq)
	if L == 0:
		result.append(rank_init)
		return result
	elif L == 1:


		k0 = int(min(rank_init, K*1))
		k1 = K - k0
		result.append(k0)
		result.append(k1)
	elif L == 2:
		k0 = int(min(rank_init, K*1))
		k1 = int(K*0.6-k0)
		result.append(k0)
		result.append(k1)

		# for l in range(2,L):
		# 	result.append(div_parts)

		result.append(K-sum(result))

	elif L == 3:
		k0 = int(min(rank_init, K*1))
		k1 = int(max(k0, K*0.9))
		result.append(k0)
		result.append(k1)
		k2 = int(K*1 - k0-k1)
		result.append(k2)

		# for l in range(2,L):
		# 	result.append(div_parts)

		result.append(K-sum(result))
	else:
		sys.exit('Not considered yet..')

	return result
	



if __name__ == '__main__':
	
	if len(sys.argv) != 5:
		sys.exit('usage: stats_edges.py <input_file_path> <input_gt_file_path> <output_file_path> <dim>')


	# assume the graph is undirected, unweighted
	weighted = True
	directed = True

	# input_file_path = sys.argv[1]
	cur_file_path = str(Path().resolve().parent)

	input_file_path = sys.argv[1]
	input_gt_path = sys.argv[2]
	output_file_path = sys.argv[3]
	K = int(sys.argv[4])

	# input_file_path = cur_file_path + '/real_graphs/facebook_combined_wei_sample_0.6.tsv'
	# input_gt_path = cur_file_path + '/real_graphs/facebook_combined_cat.tsv'

	# input_file_path = cur_file_path + '/real_graphs/digg_wei_sample_0.6.tsv'
	# input_gt_path = cur_file_path + '/real_graphs/digg_cat.tsv'
	# input_file_path = cur_file_path + '/real_graphs/dbpediad_wei_sample_0.6.tsv'
	# input_gt_path = cur_file_path + '/real_graphs/dbpediad_cat.tsv'
	# input_file_path = cur_file_path + '/real_graphs/bibsonomy_wei_sample_0.6.tsv'
	# input_gt_path = cur_file_path + '/real_graphs/bibsonomy_cat.tsv'
	# input_file_path = cur_file_path + '/real_graphs/yahoo-msg-heter/yahoo-msg-heter_wei_sample_0.6.tsv'
	# input_gt_path = cur_file_path + '/real_graphs/yahoo-msg-heter/yahoo-msg-heter_cat.tsv'
	# input_file_path = cur_file_path + '/toy_graphs/test2.tsv'
	# input_gt_path = cur_file_path + '/toy_graphs/test2_cat.tsv'

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

	raw = np.genfromtxt(input_file_path, dtype=int)
	rows = raw[:,0]
	cols = raw[:,1]
	# weis = raw[:,2]
	weis = np.ones(len(rows))


	check_eq = True
	max_id = int(max(max(rows), max(cols)))
	num_nodes = max_id + 1
	print '[max_node_id] ' + str(max_id)
	print '[num_nodes] ' + str(num_nodes)

	if max(rows) != max(cols):
		rows = np.append(rows,max(max(rows), max(cols)))
		cols = np.append(cols,max(max(rows), max(cols)))
		weis = np.append(weis, 0)
		check_eq = False


	adj_matrix = sps.lil_matrix( sps.csc_matrix((weis, (rows, cols))))
	print 'shape of adj_matrix: ' + str(adj_matrix.shape)

	# tt = adj_matrix.getrow(2)
	# print tt.toarray()
	# print tt.count_nonzero()
	# print tt.nonzero()[1]
	# temp= adj_matrix.sum(axis=0).transpose() +  adj_matrix.sum(axis=1)
	# print temp.ravel()

	# nx_graph = nx.from_scipy_sparse_matrix(adj_matrix, create_using=nx.DiGraph())
	print '--------'
	CAT_DICT, unique_cat, ID_CAT_DICT = construct_cat(input_gt_path, delimiter)


	######################################################
	# Parameters to tune
	######################################################

	nodes_to_embed = range(int(max_id)+1) #[1,2]#
	L = 2
	dim = K
	base_features = ['row', 'col', 'row_col']
	num_buckets = 2
	normalize = False
	sampling = False
	# np.set_printoptions(threshold='nan')

	######################################################

	g_sums = []

	neighbor_list = construct_neighbor_list(adj_matrix, nodes_to_embed)

	neighbor_list_r = construct_neighbor_list(adj_matrix.T, nodes_to_embed)

	graph = Graph(adj_matrix = adj_matrix, max_id = max_id, num_nodes = num_nodes, base_features = base_features, weighted = weighted,
		neighbor_list = neighbor_list,
		directed = directed, cat_dict = CAT_DICT, id_cat_dict = ID_CAT_DICT, unique_cat = unique_cat, check_eq = check_eq)

	init_feature_matrix = get_init_features(graph, base_features, nodes_to_embed)

	rep_method = RepMethod(method = "hetero", bucket_max_value = 30, num_buckets = num_buckets, alpha = 0.1, normalize = normalize)

	######################################################
	# Step 1: get base features
	######################################################


	############################################
	# The number of layers we want to explore - 
	# layer 0 is the base feature matrix
	# layer 1+: are the layers of higher order
	############################################
	# np.set_printoptions(threshold=np.nan)
	print ':('
	init_feature_matrix_seq = get_seq_features(graph, rep_method, input_dense_matrix = init_feature_matrix, nodes_to_embed = nodes_to_embed)
	# rank_init = np.linalg.matrix_rank(init_feature_matrix_seq)

	Kis = get_Kis(init_feature_matrix_seq, dim, L)
	print Kis

	feature_matrix_emb, g_sum = feature_layer_evaluation_embedding(graph, rep_method, feature_matrix = init_feature_matrix_seq, k = Kis[0])

	print '----------------------------------------'
	print 'init_feature_matrix_emb size: ' + str(feature_matrix_emb.shape)
	print 'init_g_sum size: ' + str(g_sum.shape)
	print '----------------------------------------'

	g_sums.append(g_sum)

	if rep_method.normalize:
		rep_normalized = row_normalizer(feature_matrix_emb)
	else:
		rep_normalized = feature_matrix_emb


	feature_matrix = init_feature_matrix

	for i in range(L):
		print 'Current layer: ' + str(i)
		print 'feature_matrix shape: ' + str(feature_matrix.shape)

		feature_matrix_new = search_feature_layer(graph, rep_method, base_feature_matrix = feature_matrix)

		print 'feature_matrix_new shape: ' + str(feature_matrix_new.shape)

		feature_matrix_new_seq = get_seq_features(graph, rep_method, input_dense_matrix = feature_matrix_new, nodes_to_embed = nodes_to_embed)

		feature_matrix_new_emb, g_new_sum = feature_layer_evaluation_embedding(graph, rep_method, feature_matrix = feature_matrix_new_seq, k = Kis[i+1])

		print '----------------------------------------'
		print 'feature_matrix_new_emb size: ' + str(feature_matrix_new_emb.shape)
		print 'g_new_sum size: ' + str(g_new_sum.shape)
		print '----------------------------------------'

		if rep_method.normalize:
			rep_normalized_new = row_normalizer(feature_matrix_new_emb)
		else:
			rep_normalized_new = feature_matrix_new_emb

		rep_normalized = np.concatenate((rep_normalized, rep_normalized_new), axis=1)

		g_sums.append(g_new_sum)

		feature_matrix = feature_matrix_new

	# print 'Embedding shape: ' + str(feature_matrix_emb.shape)
	# return rep_normalized


	######################################################
	# Step 2: feature proliferation
	######################################################

	print '-------------'
	for ele in g_sums:
		print ele.shape

	# np.savetxt(output_file_path, rep_normalized.round(6), fmt='%.6f', delimiter = '\t')

	N, K = rep_normalized.shape

	fOut = open(output_file_path, 'w')
	fOut.write(str(N) + ' ' + str(K) + '\n')

	for i in range(N):
		cur_line = ' '.join([str(ii) for ii in rep_normalized[i,:]])
		fOut.write(str(i) + ' ' + cur_line + '\n')

	fOut.close()

	fOut = open('g_sums.pkl', 'wb')
	pickle.dump(g_sums, fOut, -1)
	fOut.close()

	# np.savetxt('rep.tsv', rep_normalized, delimiter = '\t')
	

	viz = False
	if viz:
		tsne = TSNE(n_components=2)

		emb_2d = rep_normalized[:,0:2]
			#emb2_2d = emb2

		plt.scatter(emb_2d[:,0].T, emb_2d[:,1].T)
		for i in nodes_to_embed:
			plt.annotate(str(i), (emb_2d[i,0], emb_2d[i,1]))
		plt.show()






