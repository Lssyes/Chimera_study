import ray
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ray.init(address='auto')  # 连接到一个已存在的集群，或者用 'ray.init()' 在本地启动一个新的集群
#model=bert-base
model="bert-large"
#pipeline='gpipe'
#pipeline='1f1b'
pipeline='chimera'
#pipeline='interleave'

stages=4
ngpus=4
gpusPernode=2
nnodes=2
assert stages == ngpus
assert ngpus / gpusPernode == nnodes

profile = True

microbs=8
acc=1

# 定义使用 GPU 的任务配置
task_config = {
    "num_gpus": gpusPernode,
    "env_vars": {
        "MASTER_ADDR": "192.168.0.5",  # 根据实际环境可能需要调整
        "NCCL_SOCKET_IFNAME" : "eth0",
        
        "NSYS_NODE_INTERVAL": "1" ,  # 根据您的配置需要调整
        "NSYS_OUTPUT": f"/workspace/Chimera/bert_prof/tmp/{model}_{pipeline}_{stages}stages_{ngpus}gpus_microbs{microbs}_acc{acc}",  # 根据您的配置需要调整
        
       
        "LOCAL_SIZE" : str(gpusPernode),
        "WORLD_SIZE" : str(ngpus),
        
        
        "NODE_ID": str(-1),
        "LOCAL_RANK": str(-1),
        "RANK": str(-1),
    }
}


import subprocess
import os


# 定义一个 Ray 任务
@ray.remote(num_gpus=task_config["num_gpus"])
def train_bert(config):
    

    # 运行训练脚本 假设每个 node 都 gpusPernode个gpu
    for i in range(int(config['env_vars']["LOCAL_SIZE"])):
        config['env_vars']["LOCAL_RANK"] = str(i)
        config['env_vars']["RANK"] = str(int(config['env_vars']["NODE_ID"]) * gpusPernode + i)
        
        
        profile_str = "--profile" if profile else "" 
        train_cmd = f"""
                    echo "Starting rank-{config['env_vars']["RANK"]} script at $(date)"

                    nohup sh scripts/nsys_wrap.sh \\
                    /opt/conda/bin/python -u ./main_bert.py \\
                        --num_stages {stages} \\
                        --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \\
                        --vocab_path ./bert_data/bert-large-uncased-vocab.txt \\
                        --corpus_lines 10000 \\
                        --do_lower_case \\
                        --bert_config_path ./configs/bert_config_{model}-uncased.json \\
                        --max_seq_length 128 \\
                        --micro_batch_size {microbs} \\
                        --num_optimization_steps 3 \\
                        --gradient_accumulation_steps {acc} \\
                        --pipeline_method {pipeline} \\
                        --p2p_backend 'gloo' \\
                        --collective_backend 'nccl' \\
                        {profile_str} \\
                        --chunks 2 \\
                        --num_pipelines 2 > ./logs/{model}_{pipeline}_{stages}stages_{ngpus}gpus_microbs{microbs}_acc{acc}/{model}_{pipeline}_{stages}stages_{ngpus}gpus_microbs{microbs}_acc{acc}_node{config["env_vars"]["NODE_ID"]}_rank{config["env_vars"]["RANK"]}.log 2>&1 &
                    """
        subprocess.run(train_cmd, shell=True, env=config['env_vars'], cwd="/workspace/Chimera")
        # subprocess.run(train_cmd, shell=True, env=config['env_vars'])






pre_cmd = f"""
            rm -rf /workspace/Chimera/logs/{model}_{pipeline}_{stages}stages_{ngpus}gpus_microbs{microbs}_acc{acc}  > null.log  2>&1
            mkdir -p /workspace/Chimera/logs/{model}_{pipeline}_{stages}stages_{ngpus}gpus_microbs{microbs}_acc{acc}  > null.log  2>&1

            
            
            ssh root@192.168.0.4 -p 10022 -i ~/.ssh/HPDCLab.pem \\
                'rm -rf /workspace/Chimera/logs/{model}_{pipeline}_{stages}stages_{ngpus}gpus_microbs{microbs}_acc{acc}'
            ssh root@192.168.0.4 -p 10022 -i ~/.ssh/HPDCLab.pem \\
                'mkdir -p /workspace/Chimera/logs/{model}_{pipeline}_{stages}stages_{ngpus}gpus_microbs{microbs}_acc{acc}'                
                
            ## 代码同步
            nohup scp -r -P 10022 -i ~/.ssh/HPDCLab.pem \\
                    /workspace/Chimera/scripts root@192.168.0.4:/workspace/Chimera/ > \\
                    /workspace/Chimera/logs/scp.log 2>&1 &
                    
            nohup scp -r -P 10022 -i ~/.ssh/HPDCLab.pem \\
                    /workspace/Chimera/pipeline.py root@192.168.0.4:/workspace/Chimera/ > \\
                    /workspace/Chimera/logs/scp.log 2>&1 &
                    
            nohup scp -r -P 10022 -i ~/.ssh/HPDCLab.pem \\
                    /workspace/Chimera/main_bert.py root@192.168.0.4:/workspace/Chimera/ > \\
                    /workspace/Chimera/logs/scp.log 2>&1 &
                    
            """
profile_pro_cmd = f"""
            rm -rf {task_config['env_vars']['NSYS_OUTPUT']}  > null.log  2>&1
            mkdir -p {task_config['env_vars']['NSYS_OUTPUT']}  > null.log  2>&1
            
            ssh root@192.168.0.4 -p 10022 -i ~/.ssh/HPDCLab.pem \\
                'rm -rf {task_config['env_vars']['NSYS_OUTPUT']}  > null.log  2>&1'
            ssh root@192.168.0.4 -p 10022 -i ~/.ssh/HPDCLab.pem \\
                'mkdir -p {task_config['env_vars']['NSYS_OUTPUT']}  > null.log  2>&1'      
"""            

subprocess.run(pre_cmd, shell=True, cwd="/workspace/Chimera")
if profile:
    subprocess.run(profile_pro_cmd, shell=True, cwd="/workspace/Chimera")


print("pre_cmd done")

# 调度任务到 Ray 集群
for node_id in range(nnodes):
    task_config['env_vars']["NODE_ID"] = str(node_id)
    if node_id == 0:
        train_bert.options(resources={"ip5": 1}).remote(task_config)
    else:
        train_bert.options(resources={"ip4": 1}).remote(task_config)

import time

time.sleep(3)
    
    
    
    
