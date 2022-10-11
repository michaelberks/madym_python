# Build the image by running:
# docker build -t registry.gitlab.com/manchester_qbi/manchester_qbi_public/qbipy/qbipy_depends .
# Then push to GitLab's container registry using
# docker push registry.gitlab.com/manchester_qbi/manchester_qbi_public/qbipy/qbipy_depends
FROM python:3.7

#Install downloaded depends
ENV DEBIAN_FRONTEND noninteractive
RUN apt update \
    && apt install -y libgl1-mesa-glx \
    && pip install pytest pytest-cov gitpython numpy scipy scikit-image PyQt5 pdoc3==0.8.1 nibabel matplotlib configargparse

