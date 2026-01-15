# Prediction_RT

### Class-wise chromatographic retention time prediction using molecular fingerprints

## Overview

This repository contains the source code,  and auxiliary resources developed for the Bachelor’s Thesis “RT prediction with API for training set selection”.

The sources employed can be obtained through the following links:

all_classified.tsv --> https://drive.google.com/file/d/1Tl0Vf2o8UsSBEVw8lGJbXi63NRYZr1r0/view?usp=sharing 

0186_rtdata.tsv --> https://github.com/michaelwitting/RepoRT/tree/master/raw_data/0186

compoundsdb_2025.db --> https://drive.google.com/file/d/1W_7BjoAqTjsRd_BNuMbGyElljoKUqr0g/view?usp=sharing 

The project investigates the viability of predicting chromatographic retention time (RT) from molecular structure using fingerprint-based deep learning models, with particular emphasis on class-wise modelling strategies applied to chemically defined groups.

The repository is organised to ensure full reproducibility of the experiments described in the thesis.

## API and Docker Deployment

A prediction API is included to demonstrate the practical deployment of the proposed methodology.

The API is packaged using Docker, allowing execution without installing the full project environment. The link to download said Docker is bellow

Docker (.tar) --> https://drive.google.com/file/d/1-S9mFvArvjq49n2WCaCg0aX-JQiif_q0/view?usp=sharing

link to open API in browser: http://127.0.0.1:8000/docs

**WARNING:** The docker contains a simplified version of the final model due to emerging errors when downloading "lambda" from the final model.

Instructions for building and running the API with the final model are provided through comments in the api/api_main.py
