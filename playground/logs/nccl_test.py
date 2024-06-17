import time
import torch
import ray
import os
import torch.distributed as dist
import torch.distributed
import numpy as np
from typing import List
from torch.cuda import nvtx


DEFAULT_MASTER_ADDR = '192.168.0.5'
DEFAULT_MASTER_PORT = '1234'

FROM_RANK = 0 # ip5的第2个GPU
TO_RANK = 1   # ip4的第1个GPU

GB = 1 << 30
MB = 1 << 20

def do_all_reduce(tensor):
    
    print(f"all_reduce tensor size: {tensor.size()}")
    print(f"all_reduce tensor device: {tensor.device}")
    print(f"all_reduce tensor before reduce: {tensor}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor += torch.distributed.get_rank()
    dist.barrier()
    
    print(f"all_reduce tensor after reduce: {tensor}")

def do_send_recv(buf: torch.Tensor, 
                 is_sender: bool, 
                 is_recver: bool,
                 is_async: bool,
                 src: int = None, 
                 dst: int = None) -> None:
    req_requests = []
    
    if is_sender:
        if is_async:
            req_requests.append(dist.isend(buf, dst))
        else:
            dist.send(buf, dst)
    elif is_recver:
        if is_async:
            req_requests.append(dist.irecv(buf, src))
        else:
            dist.recv(buf, src)
    else:
        print(f"None of My bussiness {dist.get_rank()}")
        
    for req in req_requests:
        req.wait()
        
        
def do_send_recv_full_duplex(buf: List[torch.Tensor], 
                             device_group: list = None,
                             is_async: bool = None) -> None:
    assert len(buf) == 2
    assert len(device_group) == 2
    print(f"self.world_rank: {dist.get_rank()}, group: {device_group}")
    
    
    
    if is_async:
        req_requests = []
        if dist.get_rank() == device_group[0]:
            print("before isend")
            req_requests.append(dist.isend(buf[0], dst=device_group[1]))
            print("before irecv")
            req_requests.append(dist.irecv(buf[1], src=device_group[1]))
            print("after irecv")
        elif dist.get_rank() == device_group[1]:
            print("before isend")
            req_requests.append(dist.isend(buf[1], dst=device_group[0]))
            print("before irecv")
            req_requests.append(dist.irecv(buf[0], src=device_group[0]))
            print("after irecv")
        else:
            print(f"None of My bussiness do_send_recv_full_duplex {dist.get_rank()}")
            
        for req in req_requests:
            req.wait()
        
    else:
        if dist.get_rank() == device_group[0]:
            dist.send(buf[0], dst=device_group[1])
            dist.recv(buf[1], src=device_group[1])
        elif dist.get_rank() == device_group[1]:
            dist.recv(buf[0], src=device_group[0])
            dist.send(buf[1], dst=device_group[0])
        else:
            print(f"None of My bussiness do_send_recv_full_duplex {dist.get_rank()}")
    
    
    
    


def benchmark(func, repeat_num, other_func_args):
    warmup = 2 if repeat_num >= 5 else 1
    buf_sync = torch.ones(1, dtype=torch.float32).to(torch.device('cuda:0'))
    print(f"\n----------\nWarmup number: {warmup}")
    
    dist.barrier()
    for i in range(warmup):
        func(*other_func_args)
        
        
        
    print(f"Repeat {repeat_num} Benchmark")
    tic = time.time()
    
    dist.all_reduce(buf_sync)
    buf_sync = buf_sync + 1
    dist.barrier()
    for _ in range(repeat_num):
        func(*other_func_args)
        
    dist.barrier()

    
    e2e_latency = (time.time() - tic) / repeat_num
    return e2e_latency

@ray.remote(num_gpus=1)
class gpu_Actor:
    def __init__(self, local_rank, world_rank, node_id, local_size, world_size):
        nvtx.range_push('init_gpu_actor!!! -- ' + str(world_rank))
        
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.node_id = node_id
        self.local_size = local_size
        self.world_size = world_size
        os.environ["NCCL_DEBUG"] = "INFO"
        # Redirect output to a log file
        self.log_file = f'/workspace/Chimera/playground/logs/gpu_actor_{self.world_rank}.log'
        self.log_fh = open(self.log_file, 'w', buffering=1)
        self.orig_stdout = os.dup(1)
        os.dup2(self.log_fh.fileno(), 1)


        # print_world_rank
        print(f"Actor locak-{self.local_rank}/world-{self.world_rank}")
        # print(torch.cuda.device_count())
        self.device = torch.device(f"cuda:0")
        self.tensor = torch.ones(2).to(self.device) * self.world_rank
        torch.cuda.set_device(self.device)
        nvtx.range_pop()

    @property
    def is_master(self) -> bool:
        return self.world_rank == 0

        
    def init_dist_process_group(self, backend='nccl', is_high_priority=True):


        assert dist.is_available()
        master_addr = os.environ.get('MASTER_ADDR', DEFAULT_MASTER_ADDR)
        master_port = os.environ.get('MASTER_PORT', DEFAULT_MASTER_PORT)
        init_method = 'tcp://' + master_addr + ':' + master_port
        # print(init_method)
        if backend == 'nccl' and is_high_priority:
            pg_options = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        else:
            pg_options = None
        dist.init_process_group(backend,
                                init_method=init_method,
                                rank=self.world_rank,
                                world_size=self.world_size,
                                pg_options=pg_options)
        assert dist.get_rank() == self.world_rank
        assert dist.get_world_size() == self.world_size
        dist.barrier()
        # print(f"Actor locak-{self.local_rank}/world-{self.world_rank} initialized")

        return self.local_rank, self.local_size, self.world_rank, self.world_size

    def profile_full_duplex_communication(self, size, dtype, groups):
        FULL_DUPLEX_GROUP = [FROM_RANK, TO_RANK]
        is_async = False
        
        buf_1 = torch.ones(size, dtype=dtype).to(self.device) * self.world_rank
        buf_2 = torch.ones(size, dtype=dtype).to(self.device) * self.world_rank
        nBytes = buf_1.numel() * buf_1.element_size()
        
        
        profile_result = benchmark(do_send_recv_full_duplex, 5, ([buf_1, buf_2], [FROM_RANK, TO_RANK], is_async))
        
        
        if True:
            time_cost = profile_result
            array_size = nBytes
            communication_size = array_size
            bandwidth = communication_size / time_cost
            
            print(f"由于使用同步通信，所以导致通信带宽减半")
            print(f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")
            return f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\tTime: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"    
        
        pass
    
    def profile_simplex_communication(self, size, dtype, groups):
        is_sender = (self.world_rank == FROM_RANK)
        is_recver = (self.world_rank == TO_RANK)
        is_async = True
        
        
        
        buf = torch.randint(0, 1, (size,), dtype=dtype).to(self.device) * self.world_rank
        nBytes = buf.numel() * buf.element_size()

        
        number = min(max(10, int((1 << 30) / nBytes)), 1 << 13)
        

        nvtx.range_push(f"profile_simplex_communication {self.world_rank} - size: {size/GB:.5f} GB")
        profile_result = benchmark(do_send_recv, number, (buf, is_sender, is_recver, is_async, FROM_RANK, TO_RANK))
        nvtx.range_pop()
        # do_send_recv(buf, is_sender, is_async=False, src=FROM_RANK, dst=TO_RANK)
        
        if self.world_rank == FROM_RANK:
            time_cost = profile_result
            array_size = nBytes
            communication_size = array_size
            bandwidth = communication_size / time_cost
            print(f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")
            return f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\tTime: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"       
        
        
        
    def profile(self):

        
        # Benchmark simplex communication
        for sh in range(10, 29):
            print("\n______偏移量:", sh,"______")
            self.profile_simplex_communication(1<<sh, torch.float32, groups=[FROM_RANK, TO_RANK])
        
        print("\n______END_____")




        # Benchmark full-duplex communication
        # self.profile_full_duplex_communication(1<<26, torch.float32, groups=[FROM_RANK, TO_RANK])

















NODE_NUM = 2

def test_nccl_communication():

    
    GPUs = []
    GPUs.append(gpu_Actor.options(resources={"ip5": 1}).remote(local_rank=0, world_rank=0, node_id=0, local_size=2, world_size=4))
    GPUs.append(gpu_Actor.options(resources={"ip5": 1}).remote(local_rank=1, world_rank=1, node_id=0, local_size=2, world_size=4))
    GPUs.append(gpu_Actor.options(resources={"ip4": 1}).remote(local_rank=0, world_rank=2, node_id=1, local_size=2, world_size=4))
    GPUs.append(gpu_Actor.options(resources={"ip4": 1}).remote(local_rank=1, world_rank=3, node_id=1, local_size=2, world_size=4))

    # Fetch and print actor logs
    init_refs = []
    for i, gpu_actor in enumerate(GPUs):
        print(f"init gpu {i}")
        init_refs.append(gpu_actor.init_dist_process_group.remote()) 
    

    for i, gpu_init_ref in enumerate(init_refs):
        print(f"Actor {i} initialized with return value {ray.get(gpu_init_ref)}")
    
    from datetime import datetime
    print("开始 profile:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ray.get([gpu.profile.remote() for gpu in GPUs])
    print("结束 profile:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # send_task = send_tensor.options(resources={"ip5": 1}).remote(tensor, 0)
    # receive_task = receive_tensor.options(resources={"ip4": 1}).remote(tensor_size, 1)

    # start_time = time.time()
    # ray.get([send_task, receive_task])
    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # rate = tensor.numel() * tensor.element_size() * 8 / (elapsed_time * 1e9)  # Gbps
    # print(f"Transfer rate: {rate:.2f} Gbps")


if __name__ == "__main__":
    # os.environ['RAY_DEDUP_LOGS'] = '0'
    ray.init(address="auto")


    test_nccl_communication()  # Adjust tensor size as needed

    ray.shutdown()
