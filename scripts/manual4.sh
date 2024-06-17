
## bash0
## stage 0
# export CUDA_LAUNCH_BLOCKING=1


export NCCL_SOCKET_IFNAME=eth0
export MASTER_ADDR=192.168.0.5
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=4
export LOCAL_SIZE=2
python ./main_bert.py \
        --num_stages 4 \
        --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
        --corpus_lines 10000 \
        --do_lower_case \
        --bert_config_path ./configs/bert_config_bert-large-uncased.json \
        --max_seq_length 128 \
        --micro_batch_size 4 \
        --num_optimization_steps 8 \
        --gradient_accumulation_steps 1 \
        --pipeline_method chimera \
        --p2p_backend 'gloo' \
        --collective_backend 'nccl' \
        --chunks 2 \
        --num_pipelines 2


## bash1
## stage 1
# export CUDA_LAUNCH_BLOCKING=1


export NCCL_SOCKET_IFNAME=eth0
export MASTER_ADDR=192.168.0.5
export LOCAL_RANK=1
export RANK=1
export WORLD_SIZE=4
export LOCAL_SIZE=2
python ./main_bert.py \
        --num_stages 4 \
        --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
        --corpus_lines 10000 \
        --do_lower_case \
        --bert_config_path ./configs/bert_config_bert-large-uncased.json \
        --max_seq_length 128 \
        --micro_batch_size 4 \
        --num_optimization_steps 8 \
        --gradient_accumulation_steps 1 \
        --pipeline_method chimera \
        --p2p_backend 'gloo' \
        --collective_backend 'nccl' \
        --chunks 2 \
        --num_pipelines 2




## bash2
## stage 2
# export CUDA_LAUNCH_BLOCKING=1


export NCCL_SOCKET_IFNAME=eth0
export MASTER_ADDR=192.168.0.5
export LOCAL_RANK=0
export RANK=2
export WORLD_SIZE=4
export LOCAL_SIZE=2
python ./main_bert.py \
        --num_stages 4 \
        --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
        --corpus_lines 10000 \
        --do_lower_case \
        --bert_config_path ./configs/bert_config_bert-large-uncased.json \
        --max_seq_length 128 \
        --micro_batch_size 4 \
        --num_optimization_steps 8 \
        --gradient_accumulation_steps 1 \
        --pipeline_method chimera \
        --p2p_backend 'gloo' \
        --collective_backend 'nccl' \
        --chunks 2 \
        --num_pipelines 2



## bash3
## stage 3
# export CUDA_LAUNCH_BLOCKING=1


export NCCL_SOCKET_IFNAME=eth0
export MASTER_ADDR=192.168.0.5
export LOCAL_RANK=1
export RANK=3
export WORLD_SIZE=4
export LOCAL_SIZE=2
python ./main_bert.py \
        --num_stages 4 \
        --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
        --corpus_lines 10000 \
        --do_lower_case \
        --bert_config_path ./configs/bert_config_bert-large-uncased.json \
        --max_seq_length 128 \
        --micro_batch_size 4 \
        --num_optimization_steps 8 \
        --gradient_accumulation_steps 1 \
        --pipeline_method chimera \
        --p2p_backend 'gloo' \
        --collective_backend 'nccl' \
        --chunks 2 \
        --num_pipelines 2


## [g.device:] cuda:0
# [g.device:] cuda:0
# [g.device:] cuda:0
# [g.device:] cuda:0
# [g.device:] cuda:0
# [g.device:] cuda:0