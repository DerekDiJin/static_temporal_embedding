# static_temporal_embedding

The code repository for the following paper:

**Paper**: Di Jin, Ryan Rossi, Sungchul Kim and Danai Koutra. On Generalizing Static Node Embedding to Dynamic Settings. The Fifteenth International Conference on Web Search and Data Mining (WSDM), Phoenix, AZ, USA, Feb. 2022.

<p align="center">
<img src="https://derekdijin.github.io/blob/master/assets/projects/overview_up_up.jpg" width="550"  alt="Framework overview">
</p>


<!-- **Citation (bibtex)**:
```
@inproceedings{node2bits-ECML19,
   author={Di Jin and Mark Heimann and Ryan A. Rossi and Danai Koutra},
   title={Node2BITS: Compact Time- and Attribute-aware Node Representations for User Stitching},
   booktitle={ECML/PKDD},
   year={2019},
   pages={22},
}

Di Jin, Mark Heimann, Ryan A. Rossi, and Danai Koutra. "Node2BITS: Compact Time- and Attribute-aware Node Representations for User Stitching." ECML/PKDD, pp. 22. 2019.
``` -->


# Code

This repository contains an example to adopt MultiLens as the static embedding approach to obtain the dynamic embeddings.

## Inputs:

MultiLens takes two files as input, the graph file and the category file. The input files are placed under ```small_graphs_lp_time/``` directory.

### Input graph file
The input graph file can be either static or temporal edge list in the following format separated by tab:
```
<src> <dst> <weight> <timestamp> (optional)
```
The edge list is assumed to be re-ordered consecutively from 0, i.e., the minimum node ID is 0, and the maximum node ID is <#node - 1>. Some preprocess files are placed under the ```preprocess/``` directory.

### Input category file
The category file is a mapping between the node ID and its type (e.g., IP, cookie, web agent) with the following format separated by tab:
```
<category> <id_initial> <id_ending>
```
To run MultiLens correctly, each subgraph file should have its corresponding category file. But this is not required for most other methods such as node2vec, struc2vec, etc.

## Usage

To run the framework based on the predefined static embedding approach, an example is given in ```main.sh```, and the command is

```
./main.sh
```
which will run 
```main_split_heuristics.py [temp_model] [graph_series] [static_method] [initial_snapshot_id] [agg_factor] [train_num] [test_num] [time_scale] [iteration_id]```

 The complete list of argumments of this script are described as follows.

- [temp_model] denotes the specific temporal network model. It supports ```<snapshot>``` and ```<trg>```. To employ the TSG model, one can run our preprocessing code to get one aggregated summary graph and then use the selected embedding approach.
- [graph_series] denotes the graph time-series. It supports ```<TS>``` (\tau) and ```<NS>```

## Output
The output dynamic embedding file is generated under the ```src/emb<iteration_id>/``` directory, where ```<iteration_id>``` denotes the specific run of experiments. The embedding file follows the general format:
```
<node_id> [<embedding_values>]
```


# Question & troubleshooting

If you encounter any problems running the code, pls feel free to contact Di Jin (dijin@umich.edu)