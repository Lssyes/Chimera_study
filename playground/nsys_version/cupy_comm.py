"""
Benchmark the communication bandwidth with Ray + NCCL.
We use the python binding cupy.nccl to call NCCL.

Usage:
  python3 profile_communication.py
"""

import argparse
import time
import os
from contextlib import nullcontext

from torch.cuda import nvtx

import cupy as cp
from cupy.cuda import nccl
import numpy as np
import ray
from datetime import datetime
import yaml
import torch
import torch.distributed as dist
from typing import List

# parser = argparse.ArgumentParser()
# parser.add_argument("--efa", action="store_true", help="Use AWS EFS on p3.24 or p4.24 instances")
# parser.add_argument("--ib", action="store_true", help="Use InfiniBand for NCCL communcation")
# parser.add_argument("--debug", action="store_true", help="Print nccl debug information")
def _type_torch_to_cupy(torch_type: torch.dtype):
    # print(torch_type)
    mappings = {
        torch.uint8: cp.cuda.nccl.NCCL_UINT8,
        torch.int32: cp.cuda.nccl.NCCL_INT32,
        torch.int: cp.cuda.nccl.NCCL_INT,
        torch.float16: cp.cuda.nccl.NCCL_FLOAT16,
        torch.float32: cp.cuda.nccl.NCCL_FLOAT32,
        torch.float64: cp.cuda.nccl.NCCL_FLOAT64,
        torch.float: cp.cuda.nccl.NCCL_FLOAT
    }
    return mappings[torch_type]


MB = 1 << 20
GB = 1 << 30







class GpuHost:
    def __init__(self, rank, world_size, local_rank, local_size):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.local_size = local_size
        self.ct = 0
        
        self.events = []
        self.default_stream = torch.cuda.default_stream()
        
        
        self.comm = {}

        
        self.send_recv_task_id = 0
        
        
        self.events = [torch.cuda.Event(blocking=True) for _ in range(10)]
        
        self.init_global_communicator_global()

    @property
    def is_master(self):
        return self.rank == 0


    def init_global_communicator_global(self):
        
        
        
        dist.init_process_group(backend='nccl', init_method="env://", 
                                world_size=self.world_size, rank=self.rank)
        
        cp.cuda.Device(self.local_rank).use()
        self.comm_group_size = self.world_size
        print("Initialize NCCLCommunicator: <GLOBAL_COMM>; rank:", self.rank)
        
        self.dist_store = dist.distributed_c10d._get_default_store()
        if self.is_master:
            cuda_id = cp.cuda.nccl.get_unique_id()
            # print(cuda_id)
            cuda_id_str = np.array(cuda_id).tobytes()
            self.dist_store.set('group-<GLOBAL_COMM>-unique-id', cuda_id_str)
            print("Master put <group-<GLOBAL_COMM>-unique-id: ", cuda_id_str[0:10], ">.")
        else:
            cuda_id_str = self.dist_store.get('group-<GLOBAL_COMM>-unique-id')
            print("Siver get <group-<GLOBAL_COMM>-unique-id: ", cuda_id_str[0:10], ">.")
            
        comm_id = tuple(np.frombuffer(cuda_id_str, dtype=int))
        self.global_comm = cp.cuda.nccl.NcclCommunicator(self.world_size, comm_id, self.rank)
            
    

        dist.barrier()
        return True
    
    def make_communicator(self, group, name=""):
        assert self.global_comm is not None, "Global communicator is not initialized."


        nvtx.range_push(f"make_communicator_{name}")

        self.comm[name] = None
        print(f"Initialize NCCLCommunicator: <{name}>; rank:", self.rank)


        if self.is_master:
            cuda_id = cp.cuda.nccl.get_unique_id()
            # print(cuda_id)
            cuda_id_str = np.array(cuda_id).tobytes()
            self.dist_store.set(f'group-<COMM-{name}>-unique-id', cuda_id_str)
            print(f"Master put <group-<COMM-{name}>-unique-id: ", cuda_id_str[0:10], ">.")
        else:
            cuda_id_str = self.dist_store.get(f'group-<COMM-{name}>-unique-id')
            print(f"Siver get <group-<COMM-{name}>-unique-id: ", cuda_id_str[0:10], ">.")



        comm_id = tuple(np.frombuffer(cuda_id_str, dtype=int))
        if self.rank in group:
            self.comm[name] = cp.cuda.nccl.NcclCommunicator(len(group), comm_id, self.rank)
        else:
            self.comm[name] = None

        self.barrier()
        nvtx.range_pop()

        return self.comm[name]
    
    
    
    @staticmethod
    def barrier():
        dist.barrier()

    def store_set(self, key, value):
        self.dist_store.set(key, value)

    def store_get(self, key):
        return self.dist_store.get(key)


    def send(self,
             tensor: torch.Tensor,
             dst: int,
             comm=None,
             stream=cp.cuda.Stream.null):
        if comm is None:
            _comm = self.global_comm
        nvtx.range_push(f"send_{self.shift}")
        _comm.send(
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            dst,
            stream.ptr
        )
        nvtx.range_pop()
        


    def recv(self,
             tensor: torch.Tensor,
             src: int,
             comm=None,
             stream=cp.cuda.Stream.null):
        # print("Recv tensor of size:", torch.numel(tensor))
        # print("mean:", torch.mean(tensor).item(), " std:", torch.std(tensor).item())
        if comm is None:
            _comm = self.global_comm
        nvtx.range_push(f"recv_{self.shift}")
        _comm.recv(
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            src,
            stream.ptr
        )
        nvtx.range_pop()



    def all_reduce(self,
                  comm: cp.cuda.nccl.NcclCommunicator,
                  tensor: torch.Tensor,
                  stream=cp.cuda.Stream.null,
                  op=cp.cuda.nccl.NCCL_SUM):

        comm.allReduce(
            tensor.data_ptr(),
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            op,
            stream.ptr
        )


    
    
    def do_send_recv(self, tensor,  from_rank, to_rank):
        ### 这里的stream要搞懂！！
        if self.rank == from_rank:
            self.send(tensor, to_rank)
            # self.default_stream.record_event(self.events[self.send_recv_task_id])
            # self.send_recv_task_id = self.send_recv_task_id + 1
            
        elif self.rank == to_rank:
            # self.default_stream.wait_event(self.events[self.send_recv_task_id])
            # self.send_recv_task_id = self.send_recv_task_id + 1
            self.recv(tensor, from_rank)
        else:
            pass
            # print("Rank-{self.rank} do nothing.")


    def do_all_reduce(self, comm, tensor, group):
        if self.rank in group:
            self.all_reduce(comm, tensor)

                
        
    def do_send_recv_nostream(self, tensor, from_rank, to_rank):
        if self.rank == from_rank:
            print(f"Send Rank-{self.rank}  [from {from_rank} -> {to_rank}].")
            self.send(tensor, to_rank)
        elif rank == to_rank:
            print(f"Recv Rank-{self.rank}  [from {from_rank} -> {to_rank}].")
            self.recv(tensor, from_rank)
        else:
            print("Rank-{self.rank} do nothing.")
        

    def profile_send_recv(self, size, shift, dtype, from_rank, to_rank):
        self.shift = shift
        if self.global_comm == None:
            assert False, "Global communicator is not initialized."
            
        buf_tensor = torch.ones(size, dtype=dtype).cuda()
        self.do_send_recv(buf_tensor, from_rank, to_rank)
        self.do_send_recv(buf_tensor, from_rank, to_rank)
        
        repeat = min(max(10, int((1 << 30) / (buf_tensor.numel() * buf_tensor.element_size()))), 1 << 13)
        print(f"Repeat: {repeat}")
        if self.rank != from_rank and self.rank != to_rank:
            print(f"Rank-{self.rank} do nothing.")
        
        self.barrier()
        nvtx.range_push(f"PROFILE_{shift}_RANK_{self.rank}_{repeat}x")
        
        tic = time.time()
        for i in range(repeat):
            nvtx.range_push(f"TASK_{shift}_{i}")
            self.do_send_recv(buf_tensor, from_rank, to_rank)
            nvtx.range_pop()
            
        self.barrier()
        toc = time.time()
        nvtx.range_pop()
        
        
        if self.rank == from_rank:
            time_cost = (toc - tic) / repeat
            array_size = buf_tensor.numel() * buf_tensor.element_size()
            # 只发送了了一个tensor
            communication_size = array_size
            bandwidth = communication_size / time_cost
            print(f"SendRecv: {from_rank} -> {to_rank} \tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")
        
        


    def profile_all_reduce(self, comm, size, shift, dtype, group):

        self.shift = shift
        
        print(f"comm: {comm}")
            
        buf_tensor = torch.ones(size, dtype=dtype).cuda()
        self.do_all_reduce(comm, buf_tensor, group)
        self.do_all_reduce(comm, buf_tensor, group)



        repeat = min(max(10, int((1 << 30) / (buf_tensor.numel() * buf_tensor.element_size()))), 1 << 13)
        print(f"Repeat: {repeat}")
        if self.rank not in group:
            print(f"Rank-{self.rank} do nothing.")

        self.barrier()
        nvtx.range_push(f"PROFILE_ALL_REDUCE_{group}_{shift}_RANK_{self.rank}_{repeat}x")
        
        tic = time.time()
        for i in range(repeat):
            nvtx.range_push(f"TASK_{size}_{i}")
            self.do_all_reduce(comm, buf_tensor, group)
            nvtx.range_pop()
            
        self.barrier()
        toc = time.time()
        nvtx.range_pop()
        
        
        if self.rank in group:
            time_cost = (toc - tic) / repeat
            array_size = buf_tensor.numel() * buf_tensor.element_size()
            # 只发送了了一个tensor
            communication_size = array_size
            bandwidth = communication_size / time_cost
            print(f"All-reduce: {group} \tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")



    def profile(self):
        # # single Send-recv
        # nvtx.range_push(f"PROFILE_rank0 -> rank1")
        # for i in [10, 14, 18, 22, 24, 26]:
        #     print(f"\nprofile_send_recv 偏移:{i}")
        #     self.profile_send_recv(1 << i, i,  torch.float32, 0, 1)
        #     print("————————————————END————————————————\n")
        # nvtx.range_pop()

        # nvtx.range_push(f"PROFILE_rank1 -> rank2")
        # for i in [10, 14, 18, 22, 24, 26]:
        #     print(f"\nprofile_send_recv 偏移:{i}")
        #     self.profile_send_recv(1 << i, i,  torch.float32, 1, 2)
        #     print("————————————————END————————————————\n")
        # nvtx.range_pop()


        nvtx.range_push(f"PROFILE_ALL_REDUCE [0, 1]")
        group = [0, 1]
        comm = self.make_communicator(group, f"ALL_REDUCE_COMM_group_{group}")
        for i in range(10, 30, 2):
            print(f"\nprofile_ALL_REDUCE 偏移:{i}")
            self.profile_all_reduce(comm, 1 << i, i, torch.float32, group)
            print("————————————————END————————————————\n")
        nvtx.range_pop()    





    




if __name__ == "__main__":
    

    

    
    # ## 从命令行参数中获取参数
    # args = parser.parse_args()
    # dict_args = vars(args)
    # if args.config is not None:
    #     dict_args.update(yaml.safe_load(open(args.config, 'r')))
        
    
    ## 从环境变量中获取参数
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    local_size = int(os.environ.get("LOCAL_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    node_id = int(os.environ.get("NODE_ID", -1))
    whether_profile = True if os.environ.get("PROFILE", False) == "1" else False
    
    print("\n\n\n\n\n\n\n_________________ enter cupy_comm.py _________________")
    print("当前主机名:", os.uname().nodename)
    print("当前时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(f"是否进行性能分析:{whether_profile}")
    
    # # Redirect output to a log file
    # log_file = f'/workspace/Chimera/playground/nsys_version/logs/rank_{rank}.log'
    # log_fh = open(log_file, 'w')
    # orig_stdout = os.dup(1)
    # os.dup2(log_fh.fileno(), 1)
    
    
    if world_size == -1 or local_size == -1 or rank == -1 or local_rank == -1 or node_id == -1:
        print("环境变量未设置")
        exit(1)
        
    print(f"world_size:{world_size}, local_size:{local_size}, rank:{rank}, local_rank:{local_rank}, node_id:{node_id}")
    
    
    
    ## 初始化GpuHost
    gpuHost = GpuHost(rank, world_size, local_rank, local_size)
    
    
    ## 进行通信性能测试
    if whether_profile:    ## --profile \\
        print("????")
        profile_context = torch.cuda.profiler.profile()
    else:
        print("!!!!!")
        profile_context = nullcontext()
    
    with profile_context:
        print(gpuHost.profile())

