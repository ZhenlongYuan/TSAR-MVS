#!
batch_name="terrace"

prog="./exe/$batch_name/gipuma"
warping="./fusibile"
mslpdir="./data/TRAIN/$batch_name/"
inputdir="data/TRAIN/$batch_name/images/"
output_dir_basename="results/$batch_name"
p_folder="data/TRAIN/$batch_name/images/dinoSR_par.txt"
scale=1
blocksize=11
iter=8
cost_gamma=10
cost_comb="best_n"
n_best=1
image_list_array=`( cd $inputdir && ls *.jpg) `
output_dir=${output_dir_basename}/

# fuse options
fusion="./Fusion"
depth_diff=0.01
normal_thresh=15
num_consistent=1
reproj_error=2
used_list=1


echo $fusion $mslpdir --num_consistent= $num_consistent --reproj_error= $reproj_error --depth_diff= $depth_diff --angle= $normal_thresh --used_list= $used_list

$fusion $mslpdir --num_consistent= $num_consistent --reproj_error= $reproj_error --depth_diff= $depth_diff --angle= $normal_thresh --used_list= $used_list
