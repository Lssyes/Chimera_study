### Data preparation
https://github.com/microsoft/AzureML-BERT/blob/master/docs/dataprep.md


Please store `wikipedia.segmented.nltk.txt` file under the `bert_data/` directory.


N个月之后，想要启动Chimera的时候，需要做的事情

[notion笔记](https://spicy-scribe-981.notion.site/Chimera-f8fb5a0e3ac14430b6a8f524f23a38f3?pvs=4)



# 手动启动

1. 单node（2GPU）
    
    分别运行 **`scripts/manual.sh`** 上的两个命令
    
2. 多node（2GPU*2）
    
    分别运行 **`scripts/manual4.sh`** 上的两个命令
    

# ray 自动运行

Fisrt 要启动 10022 结点上的 ssh 服务器，用于代码同步，和ray结点启动（这个docker缓存有问题

若要profile的话，在启动ray的文件中，将 profile设置为True

1. 单node
    
    两个都需要改 microbatch大小
    
    ```bash
    # 启动计算
    python scripts/prof_steps_ray.py
    
    # plot call_pipeline timeline
    sh scripts/plot_cuda_timeline.sh
    ```
    
2. 多node
    
    注意，只需要运行一遍的命令放在 pre_cmd 里面
    
    ```bash
    # pkill & run-ray
    sh shutdown.sh
    
    # 启动计算
    python scripts/prof_steps_ray_mutlinode.py
    
    # plot call_pipeline timeline
    sh scripts/plot_cuda_timeline_mutlinode.sh
    ```
    

# Single Byte

在 [pipeline.py](https://github.com/Lssyes/Chimera_study/blob/master/pipeline.py) 中改

1. AllReduce 
    
    把 nb_sync_grad_very_nb 函数 改回 nb_sync_grad
    
2. send/recv
    
    在 recv_comm_thread 和 send_comm_thread
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/c2fea346-45c4-42e2-a770-5c490bc7ff85/ded67c1b-38bd-4af2-a8a4-bbef08a636d8/Untitled.png)














### Installation
```
pip install -r requirements.txt
```
For training, we use `apex.optimizers.FusedLAMB` of [NVIDIA's Apex library](https://github.com/NVIDIA/apex). Please follow the [instruction](https://github.com/NVIDIA/apex#installation) for installing `apex`. 

For profiling, we use [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems). Please make sure you can execute `nsys` command.

Our scripts are intended to run through the SLURM workload manager on a GPU cluster with 1 GPU per node.

### Profiling **Chimera** with 8 stages for BERT-Large on 8 GPUs 
```
sbatch scripts/prof_steps.sh
```
```
sh scripts/plot_cuda_timeline.sh
```
output: `bert_prof/bert-large_chimera_8stages_8gpus_microbs32_acc1.pdf`



### Publication

Chimera is pulished in SC'21, **Best Paper Finalist**. See the [paper](https://dl.acm.org/doi/abs/10.1145/3458817.3476145) and the [video talk](https://dl.acm.org/doi/abs/10.1145/3458817.3476145#sec-supp) for more details. To cite our work:
```bibtex
@inproceedings{li143,
  author = {Li, Shigang and Hoefler, Torsten},
  title = {Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines},
  year = {2021},
  isbn = {9781450384421},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3458817.3476145},
  doi = {10.1145/3458817.3476145},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  articleno = {27},
  numpages = {14},
  location = {St. Louis, Missouri},
  series = {SC '21}
}

```

### License

See [LICENSE](LICENSE).
