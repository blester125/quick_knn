#!/bin/bash

CRANURL=http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz
CRANTAR=cran.tar.gz
CRANFILE=cran.all.1400

echo "Getting Cran data."
if [ ! -e $CRANFILE ]; then
    if [ ! -e $CRANTAR ]; then
        wget $CRANURL
    fi
    mkdir cran
    tar -xvzf $CRANTAR -C cran
    mv cran/$CRANFILE .
    rm -rf cran
fi

GLOVEURL=http://nlp.stanford.edu/data/glove.6B.zip
GLOVEZIP=glove.6B.zip
GLOVEFILE=glove.6B.300d.txt

echo "Getting Glove data."
if [ ! -e $GLOVEFILE ]; then
    if [ ! -e $GLOVEZIP ]; then
        wget $GLOVEURL
    fi
    unzip $GLOVEZIP -d glove
    mv glove/$GLOVEFILE .
    rm -rf glove
fi
