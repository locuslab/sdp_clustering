The main Locale algorithm is inside the src/cluster_cpu.cpp .
The code is written in C style, but it use pyTorch as an interface.
We will refactor the code soon.

We provide the docker environment to setup all the necessary packages.
To build the environment, ensure that nvidia-docker is installed. Then
	$ cd docker
	$ sh ./build.sh
	$ sh ./run.sh
will prepare all the environment needed.
Inside the docker environment, please build with package with
	$ sudo /opt/conda/bin/python setup.py install
and the experiment can be run with
	$ python exp/exp.py
by chaning the filename in exp.py .

For demonstration, we have included the Zachary's Karate club data at data/zachary.mtx .
