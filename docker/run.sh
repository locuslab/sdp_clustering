DATA_VOLUME="-v $(pwd)/..:/data"
HOME_VOLUME="-v $HOME:$HOME"
docker run --rm --privileged --runtime=nvidia -it --net=host --ipc=host ${DATA_VOLUME} ${HOME_VOLUME} -v /etc/localtime:/etc/localtime:ro cluster bash -l
