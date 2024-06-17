
## bash0
## stage 0
# export CUDA_LAUNCH_BLOCKING=1


export MASTER_ADDR=localhost
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=2
export LOCAL_SIZE=2
python ./main_bert.py \
        --num_stages 2 \
        --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
        --corpus_lines 10000 \
        --do_lower_case \
        --bert_config_path ./configs/bert_config_bert-large-uncased.json \
        --max_seq_length 128 \
        --micro_batch_size 4 \
        --num_optimization_steps 128 \
        --gradient_accumulation_steps 1 \
        --pipeline_method chimera \
        --p2p_backend 'gloo' \
        --collective_backend 'nccl' \
        --chunks 2 \
        --num_pipelines 2


## bash1
## stage 1
# export CUDA_LAUNCH_BLOCKING=1


export MASTER_ADDR=localhost
export LOCAL_RANK=1
export RANK=1
export WORLD_SIZE=2
export LOCAL_SIZE=2
python ./main_bert.py \
        --num_stages 2 \
        --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
        --corpus_lines 10000 \
        --do_lower_case \
        --bert_config_path ./configs/bert_config_bert-large-uncased.json \
        --max_seq_length 128 \
        --micro_batch_size 4 \
        --num_optimization_steps 128 \
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



python scripts/parse_nvtx_events.py \
        bert_prof/bert-large_chimera_2stages_2gpus_microbs4_acc1_rank0.sqlite \
        --pickle_path_timeline bert_prof/bert-large_chimera_2stages_2gpus_microbs4_acc1_rank0_timeline.pickle \
        --ignore_first_event \
        --main_event_indices '5,6,7' \
        --event_keywords call_forward,call_backward,cov_kron_A,cov_kron_B,inv_kron_A,inv_kron_B,precondition,reduce,gather,sync \
        --main_event_text call_pipeline