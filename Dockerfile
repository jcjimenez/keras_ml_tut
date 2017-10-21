FROM gw000/keras:2.0.8-py3-tf-cpu
LABEL maintainer="jc.jimenez@microsoft.com"
USER root
EXPOSE 8080

RUN apt-get update -y
RUN apt-get install -y \
    libopencv-dev \
    python-opencv \
    python3-tk \
    vim

RUN pip3 install opencv-python matplotlib pillow
