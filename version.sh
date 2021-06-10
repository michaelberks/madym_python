#!/bin/bash

#This script is called by semantic-release to 
#generate a new version file automatically in Gitlab CI pipeline
echo "Writing version $1"
echo "#This file is auto-generated in GitLab CI and should not be edited manually" > VERSION
echo $1 >> VERSION
git add VERSION
git commit -m "Auto version update [ci skip]"
git push
