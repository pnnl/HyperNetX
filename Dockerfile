FROM quay.io/jupyter/base-notebook:python-3.11

# Use Jupyter Notebook as the Frontend (default is Jupyter Lab)
ENV DOCKER_STACKS_JUPYTER_CMD=notebook

# Add pre-made Notebooks for examples of how to use HNX
COPY tutorials/HNXStarter.ipynb .
COPY ["tutorials/basic/Basic 1 - HNX Basics.ipynb", "."]

RUN pip install --no-cache-dir hypernetx==2.3.2 hnxwidget==0.1.1b3

# Do not require a token when starting the notebook
ENTRYPOINT ["start-notebook.py", "--IdentityProvider.token=''"]
