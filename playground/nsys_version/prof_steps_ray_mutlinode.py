import ray
import cupy as cp
from cupy.cuda import nccl ### ??????   为什么一定要import nccl
import os
# print(cp.cuda.nccl.get_version())
# ray.init(address='auto')  # 连接到一个已存在的集群，或者用 'ray.init()' 在本地启动一个新的集群




stages=4
ngpus=4
gpusPernode=2
nnodes=2

assert stages == ngpus
assert ngpus / gpusPernode == nnodes

profile = str(1)





# 定义使用 GPU 的任务配置
task_config = {
    "num_gpus": gpusPernode,
    "env_vars": {
        "MASTER_ADDR": "192.168.0.5",  # 根据实际环境可能需要调整
        "MASTER_PORT": "9956",  # 根据实际环境可能需要调整
        "NCCL_SOCKET_IFNAME" : "eth0",
        
        "NSYS_NODE_INTERVAL": "1" ,  # 根据您的配置需要调整
        "NSYS_OUTPUT": f"/workspace/Chimera/playground/nsys_version/nsys_result/",  # 根据您的配置需要调整
        
       
        "LOCAL_SIZE" : str(gpusPernode),
        "WORLD_SIZE" : str(ngpus),
        "LOCAL_RANK": str(-1),
        "NODE_ID": str(-1),
        "RANK": str(-1),
        "PROFILE" : str(profile),
        
        # "NCCL_DEBUG": "INFO", # 根据您的配置需要调整
        "NCCL_SOCKET_NTHREADS": "4",
        "NCCL_NSOCKS_PERTHREAD": "4",
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    }
}
import subprocess
import os

import datetime
# 定义一个 Ray 任务
@ray.remote(num_gpus=task_config["num_gpus"])
def train_bert(config):

    
    


    # 运行训练脚本 假设每个 node 都 gpusPernode个gpu
    for i in range(gpusPernode):
        config['env_vars']["LOCAL_RANK"] = str(i)
        config['env_vars']["RANK"] = str(int(config['env_vars']["NODE_ID"]) * gpusPernode + i)
        
        
        profile_str = "--profile" if profile else "" 
        train_cmd = f"""

                    echo "Starting rank-{config["env_vars"]["RANK"]} script at $(date)"
                    
                    
                                            
                    

                    nohup sh playground/nsys_version/nsys_wrap.sh \\
                        /opt/conda/bin/python -u /workspace/Chimera/playground/nsys_version/cupy_comm.py > \\
                        /workspace/Chimera/playground/nsys_version/logs/rank_{config['env_vars']["RANK"]}.log 2>&1 &
                    """
        subprocess.run(train_cmd, shell=True, env=config['env_vars'], cwd="/workspace/Chimera")





pre_cmd = f"""
            rm -rf /workspace/Chimera/playground/nsys_version/logs/*  > null.log  2>&1
            rm -rf /workspace/Chimera/playground/nsys_version/nsys_result/*  > null.log  2>&1
                
            ssh root@192.168.0.4 -p 10022 -i ~/.ssh/HPDCLab.pem \\
                'rm -rf /workspace/Chimera/playground > null.log  2>&1'
                
            nohup scp -r -P 10022 -i ~/.ssh/HPDCLab.pem \\
                    /workspace/Chimera/playground root@192.168.0.4:/workspace/Chimera/ > \\
                    /workspace/Chimera/playground/nsys_version/logs/scp.log 2>&1 &
            """
subprocess.run(pre_cmd, shell=True, cwd="/workspace/Chimera")
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
    
