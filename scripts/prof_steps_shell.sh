sh scripts/nsys_wrap.sh\
    python main_bert.py \
            --num_stages $stages \
            --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
            --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
            --corpus_lines 10000 \
            --do_lower_case \
            --bert_config_path ./configs/bert_config_${model}-uncased.json \
            --max_seq_length 128 \
            --micro_batch_size $microbs \
            --num_optimization_steps 8 \
            --gradient_accumulation_steps $acc \
            --pipeline_method $pipeline \
            --p2p_backend 'gloo' \
            --collective_backend 'nccl' \
            --profile \
	        --chunks 2 \
            --num_pipelines 2\