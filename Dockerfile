ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.2-stable

FROM $BASE_CONTAINER

LABEL maintainer='UC San Diego ITS/ETS <ets-consult@ucsd.edu>'

USER root 

RUN apt-get -y install htop

USER jovyan

RUN pip install --no-cache-dir statsmodels seaborn scipy pillow networkx xgboost lightgbm torchvision torchaudio geoopt gpustat 
RUN conda install -c conda-forge python-igraph leidenalg
RUN pip install --no-cache-dir scanpy autopep8 jupyterlab toml timebudget tensorboard rich torch-tb-profiler cvxopt pot tqdm geomloss 
