#!/usr/bin/python3
# coding: utf-8

from os import path

def load_path(directory_path):
    path_list = ["docs.txt", "check_index.csv", "title_StackOverflow.txt"]
    doc_path, test_path, title_path = [path.join(directory_path, p) for p in path_list]
    return doc_path, test_path, title_path

def import_test(path):
    # -----test data-----
    import pandas as pd

    test = pd.read_csv(path).iloc[:,1:].as_matrix()
    return test

def import_title(path):
    # -----title data-----
    import numpy as np

    title = []
    with open(path, 'r', encoding="utf-8") as file:
        for line in file.readlines():
            title.append(line.strip())
    return title

def import_doc(path):
    # -----doc data-----
    doc_data = []
    with open(path, 'r') as file:
        for line in file.readlines():
            doc_data.append(line.strip())
    doc_data = [i for i in filter(None, doc_data)]
    return doc_data        

def import_data(directory_path=''):
    doc_path, test_path, title_path = load_path(directory_path)
    test_data = import_test(test_path)
    title_data = import_title(title_path)
    doc_data = import_doc(doc_path)
    return test_data, title_data, doc_data

def load_stopword():
    return {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}