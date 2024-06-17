"""
Benchmark the communication bandwidth with Ray + NCCL.
We use the python binding cupy.nccl to call NCCL.

Usage:
  python3 profile_communication.py
"""

import argparse
import time
import os

import cupy as cp
from cupy.cuda import nccl
import numpy as np
import ray

MB = 1 << 20
GB = 1 << 30


def do_all_reduce(comm, in_buffer, out_buffer):
    comm.allReduce(
        in_buffer.data.ptr,
        out_buffer.data.ptr,
        in_buffer.size,
        nccl.NCCL_FLOAT32,
        0,
        cp.cuda.Stream.null.ptr,
    )


def do_all_gather(comm, in_buffer, out_buffer):
    comm.allGather(
        in_buffer.data.ptr,
        out_buffer.data.ptr,
        in_buffer.size,
        nccl.NCCL_FLOAT32,
        cp.cuda.Stream.null.ptr,
    )


def do_send_recv(comm, buf, is_sender):
    if is_sender:
        comm.send(buf.data.ptr, buf.size, nccl.NCCL_FLOAT32,
                  1, cp.cuda.Stream.null.ptr)
    else:
        comm.recv(buf.data.ptr, buf.size, nccl.NCCL_FLOAT32,
                  0, cp.cuda.Stream.null.ptr)


from typing import List


def do_send_recv_double(comm:cp.cuda.nccl.NcclCommunicator, 
                        buf: List[cp.ndarray],
                        stream: List[cp.cuda.Stream],is_left):
    event_send_complete_left = cp.cuda.Event(block=True)
    event_send_complete_right = cp.cuda.Event(block=True)
    
    if is_left:
        comm.send(buf[0].data.ptr, buf[0].size, nccl.NCCL_FLOAT32, 1, stream[0].ptr)
        # 在发送完成后记录事件
        event_send_complete_left.record(stream[0])
        # 等待事件在接收流上完成，确保发送完成前不进行接收
        stream[1].wait_event(event_send_complete_right)
        
        comm.recv(buf[1].data.ptr, buf[1].size, nccl.NCCL_FLOAT32, 1, stream[1].ptr)
    else:
        comm.send(buf[1].data.ptr, buf[1].size, nccl.NCCL_FLOAT32, 0, stream[1].ptr)
        
        event_send_complete_right.record(stream[1])
        stream[0].wait_event(event_send_complete_left)
        
        
        comm.recv(buf[0].data.ptr, buf[0].size, nccl.NCCL_FLOAT32, 0, stream[0].ptr)


@ray.remote(num_gpus=1)
class GpuHost:
    def __init__(self, rank, world_size, nccl_uuid_list):
        self.rank = rank
        self.world_size = world_size
        self.nccl_uuid_list = nccl_uuid_list
        self.ct = 0
        
        # os.environ["NCCL_DEBUG"] = "TRACE"
        # os.environ["NCCL_BUFFSIZE"] = str(4194304 * 32)  # 8MB
        os.environ["NCCL_ALGO"] = "Tree"
        
        # Redirect output to a log file
        self.log_file = f'/workspace/Chimera/playground/alpa_logs/gpu_actor_{self.rank}.log'
        self.log_fh = open(self.log_file, 'w')
        self.orig_stdout = os.dup(1)
        os.dup2(self.log_fh.fileno(), 1)




    def init_communicator(self, groups):
        
        
        if np.max(groups) >= self.world_size:
            print("init 1")
            return None
            assert False, "init 1"

        if len(set(np.ravel(groups))) < len(np.ravel(groups)):
            print(groups)
            print(np.ravel(groups))
            print(set(np.ravel(groups)))
            print("init 2")
            return None
            assert False, "init 2"


        comm = None
        for group in groups:
            nccl_uuid = self.nccl_uuid_list[self.ct]
            self.ct += 1
            print(f"self.ct--{self.ct}")
            for device_id in group:
                if self.rank == device_id:
                    assert comm is None, "init communicator is not None"
                    print(f"rank: {group.index(self.rank)}, group: {group}")
                    comm = cp.cuda.nccl.NcclCommunicator(len(group), nccl_uuid, group.index(self.rank))

        cp.cuda.Device(0).synchronize()
        return comm

    def profile_allreduce(self, size, dtype, groups):
        comm = self.init_communicator(groups)
        if comm is None:
            return

        in_buffer = cp.ones(int(size), dtype)
        out_buffer = cp.ones(int(size), dtype)

        do_all_reduce(comm, in_buffer, out_buffer)
        do_all_reduce(comm, in_buffer, out_buffer)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_all_reduce(comm, in_buffer, out_buffer)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == 0:
            num_devices = len(groups[0])
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = 2 * array_size * (num_devices - 1) / num_devices
            bandwidth = communication_size / time_cost
            print(f"AllReduce: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")
            return f"AllReduce: {groups}\tBytes: {array_size / GB:.5f} GB\tTime: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"

    def profile_allgather(self, size, dtype, groups):
        comm = self.init_communicator(groups)
        if comm is None:
            return

        in_buffer = cp.ones(int(size) // len(groups[0]), dtype)
        out_buffer = cp.ones(int(size), dtype)

        do_all_gather(comm, in_buffer, out_buffer)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_all_gather(comm, in_buffer, out_buffer)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == 0:
            num_devices = len(groups[0])
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size * (num_devices - 1) / num_devices
            bandwidth = communication_size / time_cost
            print(f"AllGather: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")
            return f"AllGather: {groups}\tBytes: {array_size / GB:.5f} GB\tTime: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"

    def profile_send_recv(self, size, dtype, from_rank, to_rank):
        groups = [[from_rank, to_rank]]
        comm = self.init_communicator(groups)
        if comm is None:
            return

        buf = cp.ones(int(size), dtype)
        do_send_recv(comm, buf, self.rank == from_rank)
        do_send_recv(comm, buf, self.rank == from_rank)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_send_recv(comm, buf, self.rank == from_rank)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == from_rank:
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size
            bandwidth = communication_size / time_cost
            print(f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")
            return f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\tTime: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"
        
        if self.rank == to_rank:
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size
            bandwidth = communication_size / time_cost
            print(f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")
            return f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\tTime: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"


    def profile_send_recv_double(self, size, dtype, group):
        groups = [group]
        comm = self.init_communicator(groups)
        if comm is None:
            return 
        
        buf = [cp.ones(int(size), dtype), cp.ones(int(size), dtype)]
    
        stream = [cp.cuda.Stream(), cp.cuda.Stream()]
        
        print("shit-0----1")
        cp.cuda.Device(0).synchronize()
        
        do_send_recv_double(comm, buf, stream, self.rank == group[0])
        do_send_recv_double(comm, buf, stream, self.rank == group[0])
        
        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        print(f"shit-num {number}")
        cp.cuda.Device(0).synchronize()
        print("shit-1")
        tic = time.time()
        for i in range(number):
            do_send_recv_double(comm, buf, stream, self.rank == group[0])
        cp.cuda.Device(0).synchronize()
        print("shit-2")
        
        toc = time.time()

        if self.rank in group:
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size
            bandwidth = communication_size / time_cost
            print(f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")
            return f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\tTime: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"
        



    def profile_multi_send_recv(self, size, dtype, groups):
        comm = self.init_communicator(groups)
        time.sleep(1)
        comm_sync = self.init_communicator([list(np.ravel(groups))])
        if comm is None or comm_sync is None:
            return

        assert all(len(group) == 2 for group in groups)

        senders = set(group[0] for group in groups)
        receivers = set(group[1] for group in groups)

        buf = cp.ones(int(size), dtype)
        buf_sync = cp.ones(1, dtype)

        do_send_recv(comm, buf, self.rank in senders)
        do_send_recv(comm, buf, self.rank in senders)
        do_all_reduce(comm_sync, buf_sync, buf_sync)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_send_recv(comm, buf, self.rank in senders)
        do_all_reduce(comm_sync, buf_sync, buf_sync)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == groups[0][0]:
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size
            bandwidth = len(groups) * communication_size / time_cost
            print(f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")
            return f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\tTime: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s"

    def profile(self):
        # All-reduce
        return_str = ""
        # for i in range(2, 27):
        #     tmp = self.profile_allreduce(1 << i, cp.float32, [list(range(self.world_size))])
        #     if tmp is not None:
        #         return_str = return_str + tmp + "\n"
        #     tmp = self.profile_allreduce(1 << i, cp.float32, [list(range(self.world_size//2))])
        #     if tmp is not None:
        #         return_str = return_str + tmp + "\n"
        # for i in range(29, 30):
        #     tmp = self.profile_allgather(1 << i, cp.float32, [list(range(self.world_size//2))])
        #     if tmp is not None:
        #         return_str = return_str + tmp + "\n"
        #     tmp = self.profile_allgather(1 << i, cp.float32, [list(range(self.world_size))])
        #     if tmp is not None:
        #         return_str = return_str + tmp + "\n"


        # single Send-recv
        for i in range(20, 27):
            print(f"偏移:{i}")
            tmp = self.profile_send_recv(1 << i, cp.float32, 0, 1)
        
        
        # for i in range(20, 27):
        #     print(f"profile_send_recv 偏移:{i}")
        #     tmp = self.profile_send_recv(1 << i, cp.float32, 1, 2)
        # print("Finished")
        
        # for i in range(18, 27):
        #     print(f"profile_send_recv_double 偏移:{i}")
        #     tmp = self.profile_send_recv_double(1 << i, cp.float32, [1, 2])
        #     print("————————————————END————————————————\n")
        
        
        
        ##?????????
        # for i in range(20, 28):
        #     self.profile_multi_send_recv(1 << i, cp.float32, [[0, 2], [3, 1]])
        #     pass
            # self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 4], [1, self.world_size - 3]])
            # self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 2], [1, self.world_size - 1]])
            # self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 4], [1, self.world_size - 3], [2, self.world_size - 2], [3, self.world_size - 1]])
            # self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 8], [1, self.world_size - 7], [2, self.world_size - 6], [3, self.world_size - 5]])
            # self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 8], [1, self.world_size - 7], [2, self.world_size - 6], [3, self.world_size - 5],
            #                                                   [4, self.world_size - 4], [5, self.world_size - 3], [6, self.world_size - 2], [7, self.world_size - 1]])


        return return_str   


    def sync(self):
        return


def main():
    ray.init(address="auto")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--efa", action="store_true",
        help="Use AWS EFS on p3.24 or p4.24 instances")
    parser.add_argument("--ib", action="store_true",
        help="Use InfiniBand for NCCL communcation")
    parser.add_argument("--debug", action="store_true",
        help="Print nccl debug information")
    args = parser.parse_args()

    num_gpus = int(ray.cluster_resources()["GPU"])

    nccl_uuid_list = [cp.cuda.nccl.get_unique_id() for _ in range(10000)]

    workers = []
    for i in range(num_gpus):
        if args.efa:
            env_vars = {
                "FI_PROVIDER": "efa",
                "FI_EFA_USE_DEVICE_RDMA": "1",
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),  # For libnccl-net.so
                "NCCL_PROTO": "simple",
            }
        elif args.ib:
            env_vars = {
                "NCCL_SOCKET_NTHREADS": "4",
                "NCCL_NSOCKS_PERTHREAD": "4",
                "NCCL_IB_HCA": "mlx5,ibp",  # Change this to align with your IB interface name
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            }
        else:
            env_vars = {
                "NCCL_SOCKET_NTHREADS": "4",
                "NCCL_NSOCKS_PERTHREAD": "4",
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            }

        if args.debug:
            env_vars["NCCL_DEBUG"] = "INFO"

        workers.append(GpuHost.options(runtime_env={"env_vars": env_vars})\
                              .remote(i, num_gpus, nccl_uuid_list))
        
            # 文件名
    filename = '/workspaces/alpa/benchmark/cupy/profile_comm.log'
    # 按指定格式打印当前时间
    from datetime import datetime

    for _ in range(1):
        
        # 获取当前时间
        current_time = datetime.now()
        return_str = ray.get([w.profile.remote() for w in workers])


            
        ray.get([w.sync.remote() for w in workers])




main()