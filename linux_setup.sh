#!/bin/bash
# setting up the entire environment from 
# fresh Ubuntu 16.04 LTS 64-bit install
sudo apt-get -y update
sudo apt-get -y upgrade

# install accelerated BLAS environment 
# openBLAS should be there by default...just in case
sudo apt-get -y install vim git build-essential libblas-dev liblapack-dev libatlas-base-dev gfortran python-dev libfreetype6-dev libxft-dev awscli gtk2-engines-pixbuf
export LAPACK=/usr/lib/liblapack.so
export ATLAS=/usr/lib/libatlas.so

# create virtual_env and install requirements
cd twitter_LDA_topic_modeling/
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
rm get-pip.py
sudo pip install virtualenv

virtualenv -p /usr/bin/python2.7 venv
source venv/bin/activate
pip install -r requirements.txt
sudo activate-global-python-argcomplete

# install pyLDAvis from github due to pandas deprecations
pip install --upgrade -git:git://github.com/bmabey/pyLDAvis.git

# check for BLAS installation
python -c 'import numpy; numpy.show_config()'

# custom stopword list using nltk brown list plus extras
printf 'd\nstopwords\nq' | python -c 'import nltk; nltk.download()'
mv patches/english ~/nltk_data/corpora/stopwords

# custom progress bar patched
cp patches/prog_class.py venv/lib/python2.7/site-packages/pyprind/prog_class.py
