# 选择适合的基础镜像，确保与 PyTorch 兼容
# FROM pytorch/pytorch:latest
# FROM ubuntu:20.04
FROM tiangolo/uwsgi-nginx-flask:python3.11

# 更新软件源并安装 tzdata
RUN apt-get update \
    && apt-get install -y tzdata \
    && rm -rf /var/lib/apt/lists/*

# 设置时区为上海时区
ENV TZ=Asia/Shanghai

# 安装 libterm-readline-gnu-perl 
RUN apt-get update \
    && apt-get install -y libterm-readline-gnu-perl \
    && rm -rf /var/lib/apt/lists/*

# 安装其它依赖包
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 设置镜像源为阿里云镜像源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# 拷贝当前项目到 /app 目录下
COPY . /app

# 设置工作目录为 /app
WORKDIR /app

# 使用国内镜像源安装 Python 依赖
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --upgrade pip \
    && pip install --user -r requirements.txt

ENV PATH="$PATH:/root/.local/bin"
# 暴露端口
EXPOSE 80

# 执行启动命令
# CMD ["python3", "houduan.py"]



# # 二开推荐阅读[如何提高项目构建效率](https://developers.weixin.qq.com/miniprogram/dev/wxcloudrun/src/scene/build/speed.html)
# # 选择基础镜像。如需更换，请到[dockerhub官方仓库](https://hub.docker.com/_/python?tab=tags)自行选择后替换。
# # 已知alpine镜像与pytorch有兼容性问题会导致构建失败，如需使用pytorch请务必按需更换基础镜像。
# #FROM alpine:3.13
# FROM pytorch/pytorch:latest


# # 容器默认时区为UTC，如需使用上海时间请启用以下时区设置命令
# # RUN apt add tzdata && cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo Asia/Shanghai > /etc/timezone

# # 使用 HTTPS 协议访问容器云调用证书安装
# RUN apt add ca-certificates

# # 安装依赖包，如需其他依赖包，请到alpine依赖包管理(https://pkgs.alpinelinux.org/packages?name=php8*imagick*&branch=v3.13)查找。
# # 选用国内镜像源以提高下载速度
# RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tencent.com/g' /etc/apt/repositories \
# # 安装python3
# && apt add --update --no-cache python3 py3-pip \
# && rm -rf /var/cache/apt/*

# # 拷贝当前项目到/app目录下（.dockerignore中文件除外）
# COPY . /app

# # 设定当前的工作目录
# WORKDIR /app

# # 安装依赖到指定的/install文件夹
# # 选用国内镜像源以提高下载速度
# RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/ \
# && pip config set global.trusted-host mirrors.cloud.tencent.com \
# && pip install --upgrade pip \
# # pip install scipy 等数学包失败，可使用 apt add py3-scipy 进行， 参考安装 https://pkgs.alpinelinux.org/packages?name=py3-scipy&branch=v3.13
# && pip install --user -r requirements.txt

# # 暴露端口。
# # 此处端口必须与「服务设置」-「流水线」以及「手动上传代码包」部署时填写的端口一致，否则会部署失败。
# EXPOSE 80

# # 执行启动命令
# # 写多行独立的CMD命令是错误写法！只有最后一行CMD命令会被执行，之前的都会被忽略，导致业务报错。
# # 请参考[Docker官方文档之CMD命令](https://docs.docker.com/engine/reference/builder/#cmd)
# CMD ["python3", "houduan.py"]