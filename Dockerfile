# Stage 1: Copy all tutorial folders into first stage
FROM quay.io/jupyter/base-notebook:python-3.11 AS builder

WORKDIR /source

COPY tutorials/basic .
COPY tutorials/advanced .
COPY tutorials/widget .

# Stage 2: Copy only the Jupyter Notebooks defined in first stage (i.e. builder)
FROM quay.io/jupyter/base-notebook:python-3.11

# Use Jupyter Notebook as the Frontend (default is Jupyter Lab)
# See https://jupyter-docker-stacks.readthedocs.io/en/latest/index.html#choosing-jupyter-frontend
ENV DOCKER_STACKS_JUPYTER_CMD=notebook

# Only copy the Jupyter notebooks
# notebooks will be copied into '/home/jovyan' in the container
COPY --from=builder /source/Basic* .
COPY --from=builder /source/Advanced* .
COPY --from=builder /source/Demo* .

# Install the latest versions of HyperNetX and HNXWidget from PyPi
RUN pip install --no-cache-dir hypernetx hnxwidget

# When starting the Jupyter server, do not require a token when accessing URL: http://localhost:8888/tree
ENTRYPOINT ["start-notebook.py", "--IdentityProvider.token=''"]
