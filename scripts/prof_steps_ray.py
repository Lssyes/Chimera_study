import ray
import logging



ray.init(address='auto')  # 连接到一个已存在的集群，或者用 'ray.init()' 在本地启动一个新的集群
#model=bert-base
model="bert-large"
#pipeline='gpipe'
#pipeline='1f1b'
pipeline='chimera'
#pipeline='interleave'
stages=2
ngpus=2
microbs=80
acc=1

# 定义使用 GPU 的任务配置
task_config = {
    "num_gpus": 1,
    "env_vars": {
        "MASTER_ADDR": "192.168.0.5",  # 根据实际环境可能需要调整
        "NCCL_SOCKET_IFNAME" : "eth0",
        
        "NSYS_NODE_INTERVAL": "1" ,  # 根据您的配置需要调整
        "NSYS_OUTPUT": f"bert_prof/{model}_{pipeline}_{stages}stages_{ngpus}gpus_microbs{microbs}_acc{acc}",  # 根据您的配置需要调整
        
        "LOCAL_RANK": str(0),
        "RANK": str(-1),
        "WORLD_SIZE" : str(ngpus),
        "LOCAL_SIZE" : str(1),
    }
}

profile = True
profile_str = "--profile" if profile else ""

# 定义一个 Ray 任务
@ray.remote(num_gpus=task_config["num_gpus"])
def train_bert(config):
    import subprocess
    import os


    # 运行训练脚本
    os.environ.update(config['env_vars'])     

    train_cmd = f"""
                echo "Starting training script at $(date)"

                nohup sh scripts/nsys_wrap.sh \\
                python ./main_bert.py \\
                    --num_stages {stages} \\
                    --corpus_path ./bert_data/wikipedia.segmented.nltk.txt.1 \\
                    --vocab_path ./bert_data/bert-large-uncased-vocab.txt \\
                    --corpus_lines 1000000 \\
                    --do_lower_case \\
                    --bert_config_path ./configs/bert_config_{model}-uncased.json \\
                    --max_seq_length 128 \\
                    --micro_batch_size {microbs} \\
                    --num_optimization_steps 8 \\
                    --gradient_accumulation_steps {acc} \\
                    --pipeline_method {pipeline} \\
                    --p2p_backend 'gloo' \\
                    --collective_backend 'nccl' \\
                    {profile_str} \\
                    --chunks 2 \\
                    --num_pipelines 2 > ./logs/{model}_{pipeline}_{stages}stages_{ngpus}gpus_microbs{microbs}_acc{acc}_{config["env_vars"]["RANK"]}.log 2>&1 &

                """
    print("train_cmd started")
    subprocess.run(train_cmd, shell=True)



# 调度任务到 Ray 集群
for stage_id in range(stages):
    task_config["env_vars"]["RANK"] = str(stage_id)
    train_bert.remote(task_config)

import time

time.sleep(3)
    
    
    
    
