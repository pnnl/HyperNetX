Brenda Praggastis, H2D final deliverable
----------------------------------------
This directory contains three items:

Tutorial_6_HNX_Static_Classes_and_NWHy_handshake.pdf
h2d-hnx-nwhy_v5.tar.gz
README.txt

The zipped file h2d-hnx-nwhy_v5.tar.gz is a dockerized container with the latest 
HNX development code and support software for using NWHypergraph (NWHy) and NWGraph  
as a backend for HNX. 

There are two new tutorials included in hypernetx.

Instructions for using the dockerized container stored in h2d-hnx-nwhy_v5.tar.gz
--------------------------------------------------------------------------------

You will need to have docker installed on your computer.
Enter the following commands:

>> gunzip h2d-hnx-nwhy_v5.tar.gz
>> docker load -i h2d-hnx-nwhy_v5.tar
>> docker run -it -p 8890:8890 h2d-hnx-nwhy:5

This will return a prompt like this:
ubuntu@28bd1d9e5296:~$

The dockerized container starts within a NWhy_bp directory. 
Change to the ubuntu home directory to access hypernetx and other available folders:

>> cd 

You can look through the filesystem or go directly to the HNX tutorials.
These are in Jupyter notebooks.
To run a jupyter notebook:

>> jupyter notebook --no-browser --ip=0.0.0.0 --port=8890. ## note the port number is the same as the port number on line 15 above

You will get output that looks like this:

    To access the notebook, open this file in a browser:
        file:///home/ubuntu/.local/share/jupyter/runtime/nbserver-135-open.html
    Or copy and paste one of these URLs:
        http://28bd1d9e5296:8890/?token=c15e594d94a..... long string
     or http://127.0.0.1:8890/?token=c15e594d94a83b.....

Copy and paste one of the http urls on the screen into your browser's address bar.

Assuming you were in the home directory to begin with, this should open a directory to the whole file system.
Jupyter provides an easy way to look at the code in all of these files just by clicking on links.
To go to the tutorials:

Click on hypernetx
Click on tutorials
There are two new tutorials.
Click on Tutorial 6 - ... - for Static Classes and HNX <-> NWHy handshake
Click on Tutorial 7 - ... - for s-centrality measures

Instructions for the tutorials are inline.
