ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.2-stable

FROM $BASE_CONTAINER

LABEL maintainer='UC San Diego ITS/ETS <ets-consult@ucsd.edu>'

USER root 

RUN apt-get -y install htop

USER jovyan

RUN pip install --no-cache-dir statsmodels seaborn scipy pillow networkx xgboost lightgbm torchvision git+https://github.com/geoopt/geoopt.git@010f5bdf1d6cf217fb47a767e2d93e39369e4cb2#egg=geoopt
