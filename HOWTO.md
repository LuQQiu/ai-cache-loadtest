HOWTO run load agent with torch distributed run

* Download demo image: lvlouisaslia/alluxioloadagent:latest
* Start docker with following command
```
docker run -it --rm --name loadtest -e NVIDIA_VISIBLE_DEVICES= -v `pwd`:/v/ -w /v lvlouisaslia/alluxioloadagent:latest bash
```

* Start load agent by

prepare filelist into inputdata.csv, one filepath per line with the common alluxio path prefix. The common path prefix will be passed to load-agent.py
```
./run-test.sh 2 load-agent.py --workers 6 --inputfile inputdata.csv --number_of_files 500000000 -P alluxio://<server>:<port>/path/of/prefix/
```
