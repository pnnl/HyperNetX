## Using Docker

**Building a file**

1. First create a docker file that you will build
-- Start in a folder with a Dockerfile and all of the references you need
-- Sample docker files exist on docker docs: http://docs.docker.com
2. docker build - < Dockerfile
3. docker build . ## this will dockerize everything in the current folder Will it read the dockerfile as well?
4. docker build --output type=tar,dest=out.tar .

**Run the docker file**

On puma you will need to ssh in `ssh -L 8197:localhost:8197 -Y puma.pnl.gov`

    docker run -it -p 8883:8883 nwhy-bp-112320:2 bash  ## you need a shell command to get things started if the tar file was exported and not saved
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port=8883

    docker ps [-a if you want old container ids]
    docker cp [OPTIONS] CONTAINER:SRC_PATH DEST_PATH|-   ###CONTAINER = CONTAINERID
    docker cp [OPTIONS] SRC_PATH|- CONTAINER:DEST_PATH

    DEST_PATH shoud start from root: /home/ubuntu/...


**from outside of the running instance**
    docker ps -a ## get container_id
    docker commit container_id repository:tag ## commits changes to image
    docker save image_to_save -o outputname.tar   ### now we can export it

    gzip tarfile for sending
    gunzip tarfile for opening

    docker import nwhy_latest.tar.gz  nwhy-hnx-112720:3  ### create a local image based on a tar file

**Run a docker image for hnx-nwhy**
docker images

    REPOSITORY           TAG                 IMAGE ID            CREATED             SIZE
    h2d-hnx-nwhy         5                   8b99d509e582        21 hours ago        2.1GB

(hnx) WE38379:~ prag717$ docker run -it -p 8877:8877 nwhy-hnx-112720:3
ubuntu@c37c3d58649e:~/NWhy_pb$ cd               
ubuntu@c37c3d58649e:~$ jupyter notebook --no-browser --ip=0.0.0.0 --port=8877

## Dev-Central requests:
https://jira.pnnl.gov/request/

/scratch/hnx/latest.tar.gz

**This worked:**
    docker run -it -p 8897:8897 nwhy-bp-112320:1 bash
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port=8897

    docker create nwhy-bp-112320:2 bash
    docker run -it -p 8897:8897 nwhy-bp-112320:2 bash

    docker load -i nwhy_latest.tar   ## If Tony saves instead of export, then load the tar file. You start with his image name and tags. 

    docker run -it -p 8897:8897 nwhy:latest bash
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port=8897

## Log into Puma
```
ssh -L 8192:localhost:8192 -Y puma.pnl.gov   
#salloc -p shared -N 1 -A <project name> #--gres=gpu:1
#ssh -L 8192:localhost:8192 -Y  ${SLURM_JOB_NODELIST}
bash
module load gcc/7.1.0
conda activate hnx 
jupyter-notebook --ip=0.0.0.0 --no-browser --port=8192

ssh -L 8192:localhost:8192 -Y puma.pnl.gov
```


ssh -L 8884:compute-node:8884 puma
ssh -L 8884:127.0.0.1:8884 puma

