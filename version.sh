#!/bin/bash

#This script is called by semantic-release to 
#generate a new version file automatically in Gitlab CI pipeline
echo "Writing version $1"
echo "#This file is auto-generated in GitLab CI and should not be edited manually\n" > VERSION.txt
echo $1 >> VERSION.txt
ls
