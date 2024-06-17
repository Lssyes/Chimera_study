pkill -9 python


# scp -r -P 10022 -i ~/.ssh/HPDCLab.pem /workspace/Chimera root@192.168.0.4:/workspace/


RAY_ROOT="/opt/conda/bin/ray"

pkill -9 python
ssh root@192.168.0.4 -p 10022 -i ~/.ssh/HPDCLab.pem pkill -9 python



${RAY_ROOT} stop
ssh root@192.168.0.4 -p 10022 -i ~/.ssh/HPDCLab.pem  "${RAY_ROOT} stop"


${RAY_ROOT} start --head  --resources='{"ip5": 2}'
ssh root@192.168.0.4 -p 10022 -i ~/.ssh/HPDCLab.pem 'ray start --address="192.168.0.5:6379" --resources="{\"ip4\": 2}"'
# export https_proxy=http://192.168.0.5:7890 http_proxy=http://192.168.0.5:7890 all_proxy=socks5://192.168.0.5:7890
# export https_proxy=http://192.168.0.9:8890 http_proxy=http://192.168.0.9:8890 all_proxy=socks5://192.168.0.9:8890

# docker run -d –name v2ray -v $(pwd)/config.json:/etc/v2ray/config.json -p 8890:8890 v2ray/official

# docker run -d --name v2ray -e TZ=Asia/Shanghai -v /etc/v2ray:/etc/v2ray -p 8890:8890 --restart always v2fly/v2fly-core run -c /etc/v2ray/config.json

# docker run --init -it -d --privileged -p 5901:5901 -p 6080:6080 -p 8022:22 --shm-size=10.24G mathworks/matlab:r2022b -vnc
 
# curl -L -s https://github.com/mzz2017/V2RayA/raw/master/install/go.sh | sudo -E bash -s - --source ustc

# nohup ./frpc -c frpc.toml > runoob.log 2>&1 &
