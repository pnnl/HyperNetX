## Using Docker

**Building a file**

1. First create a docker file that you will build
-- Start in a folder with a Dockerfile and all of the references you need
-- Sample docker files exist on docker docs: http://docs.docker.com
2. docker build - < Dockerfile
3. docker build . ## this will dockerize everything in the current folder Will it read the dockerfile as well?
4. docker build --output type=tar,dest=out.tar .

**Run the docker file**

    docker run -it -p 8891:8891 nwhy:latest
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port=8891

    docker ps [-a if you want old container ids]
    docker cp [OPTIONS] CONTAINER:SRC_PATH DEST_PATH|-   ###CONTAINER = CONTAINERID
    docker cp [OPTIONS] SRC_PATH|- CONTAINER:DEST_PATH


**from outside of the running instance**
    docker ps -a ## get container_id
    docker commit container_id ## commits changes to image
    docker save image_to_save -o outputname.tar

    gzip tarfile for sending
    gunzip tarfile for opening

**Run a docker image for hnx-nwhy**
docker images

    REPOSITORY           TAG                 IMAGE ID            CREATED             SIZE
    h2d-hnx-nwhy         5                   8b99d509e582        21 hours ago        2.1GB

(hnx) WE38379:~ prag717$ docker run -it -p 8884:8884 h2d-hnx-nwhy:5
ubuntu@c37c3d58649e:~/NWhy_pb$ cd               
ubuntu@c37c3d58649e:~$ jupyter notebook --no-browser --ip=0.0.0.0 --port=8884

## Dev-Central requests:
https://jira.pnnl.gov/request/

/scratch/hnx/latest.tar.gz

## Log into Puma
ssh -L 8192:localhost:8192 -Y puma.pnl.gov   
salloc -p shared -N 1 -A BandonDunes #--gres=gpu:1
ssh -L 8192:localhost:8192 -Y  ${SLURM_JOB_NODELIST}

ssh -L 8192:localhost:8192 -Y puma.pnl.gov
module load gcc/7.1.0
conda activate ddp 
jupyter-notebook --ip=0.0.0.0 --no-browser --port=8192




