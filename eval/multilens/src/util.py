import numpy as np, networkx as nx
import scipy.sparse

class RepMethod():
	def __init__(self, bucket_max_value=None, method="hetero", alpha = 0.1, num_buckets = None, use_other_features = False, normalize = True,
			implicit_factorization = True):
		self.method = method #representation learning method
		# self.k = k #sample 2k log N points
		self.bucket_max_value = bucket_max_value #furthest hop distance up to which to compare neighbors
		self.alpha = alpha #discount factor for higher layers
		self.num_buckets = num_buckets #number of buckets to split node feature values into #CURRENTLY BASE OF LOG SCALE
		self.use_other_features = use_other_features
		self.normalize = normalize
		self.use_mean = True
		self.use_var = True
		self.use_sum = True
		self.use_prod = False	# <<<<
		self.use_max = True
		self.use_min = True
		self.use_l1 = True
		self.use_l2 = True
		self.use_total = 7


class Graph():
	#Undirected, unweighted
	def __init__(self, adj_matrix = None, num_nodes = None, max_id = None, weighted = False, directed = False, neighbor_list = None,
			num_buckets = None, base_features = None, cat_dict = None, id_cat_dict = None, unique_cat = None, check_eq = True):
		# self.nx_graph = nx_graph
		self.adj_matrix = adj_matrix
		self.num_nodes = num_nodes
		self.max_id = max_id
		self.base_features = base_features
		self.unique_cat = unique_cat
		self.weighted = weighted
		self.directed = directed
		self.num_buckets = num_buckets #how many buckets to break node features into

		self.max_features = {}

		# for feature in self.node_features:
		# 	if node_features_dict is None:
		# 		self.max_features[feature] = 0
		# 	else:
		# 		self.max_features[feature] = np.max(self.node_features_dict[feature])
		self. neighbor_list = neighbor_list
		self.cat_dict = cat_dict
		self.id_cat_dict = id_cat_dict
		self.check_eq = check_eq
		


	# def set_node_features(self, node_features_outer):
	# 	self.node_features_dict = node_features_outer
	# 	for feature in self.node_features: #TODO hacky to handle this case separately
	# 		self.max_features[feature] = np.max(self.node_features_dict[feature])
	# 	print self.max_features
			

	# 	# if "edge_label" in features_to_compute: #can't set this for each node, since it depends on what edge it's connected to...just set max feature
	# 	# 	if self.edge_labels is not None:
	# 	# 		self.max_features["edge_label"] = np.max(self.edge_labels)
	# 	# 	else:
	# 	# 		raise ValueError("no edge labels to bin")
	# 	#print "features to compute:", features_to_compute
	# 	self.set_node_features(new_node_features)

	# def normalize_node_features(self):
	# 	normalized_features_dict = dict()
	# 	for feature in self.node_features:
	# 		normalized_features = self.node_features[feature]
			
	# 		#scale so no feature values less than 1 (for logarithmic binning)
	# 		if np.min(normalized_features) < 1:
	# 			normalized_features /= np.min(normalized_features[normalized_features != 0])
	# 			if np.max(normalized_features) == 1: #e.g. binary features
	# 				normalized_features += 1
	# 			normalized_features[normalized_features == 0] = 1 #set 0 values to 1--bin them in first bucket (smallest values)
	# 		normalized_features_dict[feature] = normalized_features
	# 	self.set_node_features(normalized_features_dict)


class Node():
	def __init__(self, node_id, centrality = None, attributes = None, parent = None, cat = None):
		self.node_id = node_id
		self.centrality = centrality
		self.attributes = attributes
		self.parent = parent
		self.cat = cat

	def set_centrality(self, centrality):
		self.centrality = centrality


	####################################################
	# Assumption: each node has only one type 
	####################################################

	def set_cat(self, cat_dict):
		for t in cat_dict.keys():
			if self.node_id in cat_dict[t]:
				self.cat = t
				break


