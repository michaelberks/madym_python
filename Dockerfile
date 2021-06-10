FROM python:3.7

#Install downloaded depends
ENV DEBIAN_FRONTEND noninteractive
RUN apt update \
    && apt install -y libgl1-mesa-glx \
    && pip install pytest pytest-cov numpy scipy scikit-image PyQt5 pdoc3==0.8.1   

