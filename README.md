# E-Commerce Text Classification Project

## Project Overview

This project is centered around the application of Natural Language Processing (NLP) to classify e-commerce products into categories based on their descriptions. It showcases a range of machine learning models and techniques to analyze and predict product categories, contributing to the automation of product listing processes.

## Dataset Description
https://doi.org/10.5281/zenodo.3355823

The dataset includes various product descriptions from an e-commerce platform. Each product description is labeled with a category that it belongs to. The categories include but are not limited to electronics, clothing, books, and household items.

## Models and Techniques

In this project, the following models and techniques are implemented:

- **Naive Bayes**: A probabilistic classifier that applies Bayes' theorem.
- **Support Vector Machines (SVM)**: A robust classifier that finds the optimal hyperplane for classification.
- **XGBoost**: An efficient implementation of gradient boosting framework.
- **BERT (Bidirectional Encoder Representations from Transformers)**: A transformer-based model pre-trained on a large corpus of text for high-quality language representations.

## Requirements

To run this project, you need the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- spacy
- transformers
- torch

## Installation

To set up the project environment, run the following commands:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk spacy transformers torch
python -m spacy download en_core_web_sm
