import torch

import ray


ray.init(address='auto')  # 连接到一个已存在的集群，或者用 'ray.init()' 在本地启动一个新的集群




# 定义一个 Ray 任务
@ray.remote(num_gpus=2)
def train_bert():
    print(torch.cuda.device_count())
    torch.cuda.set_device(0)
    return torch.cuda.current_device()
    
    # 调度任务到 Ray 集群
a = [train_bert.remote() for _ in range(1)]
print(ray.get(a))  # [0, 0]