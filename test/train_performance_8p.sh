Network="SPACH"
batch_size=128
export RANK_SIZE=8
data_path=""
max_iter=1210
model=""

# check argument
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --model* ]];then
        model=`echo ${para#*=}`
    fi
done

# check dataset path
if [[ $data_path == "" ]];then
    echo "[Error] para \"data_path\" must be confing"
    exit 1
fi

cur_path=`pwd`
cur_path_last_diename=${cur_path##*/}
if [ x"${cur_path_last_diename}" == x"test" ];then
    test_path_dir=${cur_path}
    cd ..
    cur_path=`pwd`
else
    test_path_dir=${cur_path}/test
fi
echo ${pwd}

#save training log
ASCEND_DEVICE_ID=0
if [ -d ${test_path_dir}/output/${ASCEND_DEVICE_ID} ];then
    rm -rf ${test_path_dir}/output/${ASCEND_DEVICE_ID}
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
else
    mkdir -p ${test_path_dir}/output/$ASCEND_DEVICE_ID
fi

export SPACH_DATASETS=${data_path}
export PYTHONPATH=./:$PYTHONPATH

#training star time
start_time=$(date +%s)
#source environment
check_etp_flag=`env | grep etp_running_flag`
etp_flag=`echo ${check_etp_flag#*=}`
if [ x"${etp_flag}" != x"true" ];then
    source ${test_path_dir}/env_npu.sh
    export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
fi

get_lscpu_value() {
    awk -F: "(\$1 == \"${1}\"){gsub(/ /, \"\", \$2); print \$2; found=1} END{exit found!=1}"
}

lscpu_out=$(lscpu)
n_sockets=4
n_cores_per_socket=$(get_lscpu_value 'Core(s) per socket' <<< "${lscpu_out}")

echo "num_sockets = ${n_sockets} cores_per_socket=${n_cores_per_socket}"

export PYTHONPATH=../:$PYTHONPATH

python3.7 -u -m bind_pyt \
    --nsockets_per_node ${n_sockets} \
    --ncores_per_socket ${n_cores_per_socket} \
    --master_addr $(hostname -I |awk '{print $1}') \
    --no_hyperthreads \
    --no_membind "$@" main.py \
    --model ${model} \
    --npu \
    --data-path ${data_path} \
    --pin-mem \
    --dist-eval \
    --num_workers 16 \
    --epochs 5\
    --output_dir ${test_path_dir}/output/${ASCEND_DEVICE_ID} \
    > ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &

wait

end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

echo "------------------ Final result ------------------"

FPS=`cat ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep 'FPS:'| awk '{sum+=$10} END {print sum/NR}'`
echo "Final Performance FPS : ${FPS}"
echo "E2E Training Duration sec : $e2e_time"

BatchSize=${batch_size}
DeviceType=`uname -m`
CaseName=${Network}_bs${BatchSize}_${RANK_SIZE}'p'_'acc'

AvgFPS=${FPS}

MinLoss=`cat ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep 'Averaged stats:' | awk 'BEGIN {min = 65536} {if ($12+0 < min+0) min=$12} END {print min}'`
MaxAccuracy=`cat ${test_path_dir}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log | grep 'Max accuracy' | awk 'BEGIN {max = 0} {if ($9+0 > max+0) max=$9} END {print max}'`

#key meassage in ${CaseName}.log
echo "Network = ${Network}" >  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "RankSize = ${RANK_SIZE}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "CaseName = ${CaseName}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "AvgFPS = ${AvgFPS}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "MinLoss = ${MinLoss}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "MaxAccuracy = ${MaxAccuracy}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>  ${test_path_dir}/output/$ASCEND_DEVICE_ID/${CaseName}.log
