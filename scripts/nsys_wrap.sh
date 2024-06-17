#!/bin/bash
echo "\n\n\n________________________________________________________________________________________"
echo "[Running nsys_wrap.sh rank: ${RANK}]"
echo "\$NSYS_OUTPUT -> ${NSYS_OUTPUT}"






if [[ -z "${NSYS_OUTPUT}" ]]; then
    NSYS_OUTPUT=prof
fi
if [[ -z "${NSYS_NODE_INTERVAL}" ]]; then
    NSYS_NODE_INTERVAL=1
fi

sleep 2




if [ -n "${RANK}" ];
then
    nsys profile \
        -f true \
        -o ${NSYS_OUTPUT}/_nccl_rank${RANK} \
        -c cudaProfilerApi \
        --trace cuda,nvtx,cudnn,osrt \
        --export sqlite \
    $@
else
    $@
fi
echo "Training script completed at $(date), watching for nsys to complete {30s}..."
sleep 30  # wait for nsys to complete

# if [ "$RANK" -eq 0 ]; 
# then
#     echo "LOCAL_RANK is 0, executing the scp."
#     scp -r -P 10022 -i ~/.ssh/HPDCLab.pem root@192.168.0.4:/workspace/Chimera/bert_prof/tmp/bert-large_chimera_4stages_4gpus_microbs8_acc1/*  /workspace/Chimera/bert_prof/tmp/bert-large_chimera_4stages_4gpus_microbs8_acc1/
# else
#     echo "LOCAL_RANK is not 0, skipping the scp ."
# fi


echo "nsys script completed at $(date)"

