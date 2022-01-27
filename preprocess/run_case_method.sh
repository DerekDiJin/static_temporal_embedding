#
filename=$1
splits=$2

python s4_reorder.py "$filename"_subTS_month "$splits" multilens TS _s
python s4_reorder.py "$filename"_subTS_month "$splits" multilens TS _p
python s4_reorder.py "$filename"_subNS_month "$splits" multilens NS _s
python s4_reorder.py "$filename"_subNS_month "$splits" multilens NS _p