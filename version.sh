#!/bin/bash
echo "#This file is auto-generated in GitLab CI and should not be edited manually" > src/QbiPy/VERSION
git fetch
git describe --tags --abbrev=0 >> src/QbiPy/VERSION