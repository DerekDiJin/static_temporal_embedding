#!/bin/bash

train_init=10
agg_factor=1
train_num=6
test_num=1
time_scale=month

for j in multilens #struc2vec graphwave #node2vec line struc2vec multilens role2vec graphwave g2g 
do
	for i in 1 #2 3
	do
		echo "$train_init" "$train_end" "$i" "$j"
		python main_split_heuristics.py snapshot TS "$j" "$train_init" "$agg_factor" "$train_num" "$test_num" "$time_scale" "$i"
		python main_split_heuristics.py trg TS "$j" "$train_init" "$agg_factor" "$train_num" "$test_num" "$time_scale" "$i"

		python main_split_heuristics.py snapshot NS "$j" "$train_init" "$agg_factor" "$train_num" "$test_num" "$time_scale" "$i"
		python main_split_heuristics.py trg NS "$j" "$train_init" "$agg_factor" "$train_num" "$test_num" "$time_scale" "$i"
		
		# python main_split_heuristics.py snapshot NTS "$j" "$train_init" "$agg_factor" "$train_num" "$test_num" "$time_scale" "$i"
		# python main_split_heuristics.py trg NTS "$j" "$train_init" "$agg_factor" "$train_num" "$test_num" "$time_scale" "$i"
	done
done


