#!
batch_name="terrains"

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


#warping conf
count=0
for im in $image_list_array
do
    echo $count
    image_list=( $im )
    mkdir -p $output_dir # 这里创建result文件夹
    for ij in $image_list_array
    do
	if [ $im != $ij ]
	then
	    image_list+=( $ij )
	fi
    done
	echo "im:"$im""
	echo ${image_list[@]} # image_list第一个元素为img其他为其他

    cmd="$prog ${image_list[@]} -mslp_folder $mslpdir -images_folder $inputdir -krt_file $p_folder -output_folder $output_dir -no_display --cam_scale=$scale --iterations=$iter --blocksize=$blocksize --cost_gamma=$cost_gamma --cost_comb=best_n --n_best=$n_best --min_angle=$min_angle --max_angle=$max_angle"
    echo $cmd
    $cmd
    let "count += 1"
done
echo "done gipuma"