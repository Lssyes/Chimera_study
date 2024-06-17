# nsys profile \
#  -f true \
#  -o /workspace/Chimera/playground/logs/my_profile_output \
#  -c cudaProfilerApi \
#  --trace cuda,nvtx,cudnn,osrt \
#  --export sqlite \
#  /opt/conda/bin/python /workspace/Chimera/playground/nccl_test.py



#!/bin/bash

# 定义颜色
BLUE="\033[0;34m"
NC="\033[0m" # No Color

# 使用蓝色输出
# echo -e "${BLUE}This is a blue text${NC}"

echo -e "${BLUE}[playground/nsys_version/nsys_wrap.sh rank-${RANK}]   $(date)${NC}"
echo "output:  ${NSYS_OUTPUT}_nccl_test_rank-${RANK}"



if [[ -z "${NSYS_OUTPUT}" ]]; then
    NSYS_OUTPUT=prof
fi
if [[ -z "${NSYS_NODE_INTERVAL}" ]]; then
    NSYS_NODE_INTERVAL=1
fi

sleep 2

echo ${PROFILE}

if [ "${PROFILE}" = "1" ];
then
    echo "Rank-${RANK} nsys profile"
    nsys profile \
        -f true \
        -o ${NSYS_OUTPUT}_nccl_test_rank-${RANK} \
        -c cudaProfilerApi \
        --trace cuda,nvtx,cudnn,osrt \
        --export sqlite \
    $@
else
    echo "Rank-${RANK} Run command without nsys profile"
    $@
fi


# nsys export --type=sqlite --ts-normalize=true --output=playground/nsys_version/nsys_result/_nccl_test_rank-0.sqlite playground/nsys_version/nsys_result/_nccl_test_rank-0.nsys-rep

        
# if [ "${RANK}" = "0" ];
# then
#     scp playground/nsys_version/nsys_result/* root@192.168.0.5:/dt/lssyes/downloads/
#     ssh root@192.168.0.4 -p 10022 -i ~/.ssh/HPDCLab.pem 'scp playground/nsys_version/nsys_result/* root@192.168.0.5:/dt/lssyes/downloads/'
# fi


sleep 30  # wait for nsys to complete

