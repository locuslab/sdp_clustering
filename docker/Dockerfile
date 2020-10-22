FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
ARG HOME_DIR

RUN addgroup --gid ${GROUP_ID} ${USER_NAME} || groupmod -n ${USER_NAME} $(getent group ${GROUP_ID})
RUN apt-get -q update; apt-get -q -y install sudo vim
RUN conda install -y -q jupyter matplotlib line_profiler
#RUN conda install -y -q pytorch=1.3.1 -c pytorch
RUN adduser --quiet --disabled-password --system --no-create-home --uid ${USER_ID} --gid ${GROUP_ID} --gecos '' --shell /bin/bash ${USER_NAME}
RUN usermod -d ${HOME_DIR} ${USER_NAME}
RUN adduser --quiet ${USER_NAME} sudo ; echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN pip install pytorch_memlab setproctitle termcolor
RUN conda install -y -q scipy
RUN pip install cvxpy

RUN mkdir -p /data
WORKDIR /data

USER ${USER_NAME}
