# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:46:02 2024

@author: fifon
"""


number_560 = 1.0
number_320 = 0.572
number_160 = 0.287
number_80 = 0.143
number_40 = 0.073
number_20 = 0.037
number_10 = 0.019
number_5 = 0.01

def extract_accuracies_acc(data, seed):
    percentages = [number_560, number_320, number_160, number_80, number_40, number_20, number_10, number_5]
    test_accuracies = [
        (next(item for item in data[seed] if item["train_percentage"] == percentage))["test_acc"] * 100
        for percentage in percentages
    ]
    return test_accuracies


def extract_accuracies_f1(data, seed):
    percentages = [number_560, number_320, number_160, number_80, number_40, number_20, number_10, number_5]
    test_f1s = [
        (next(item for item in data[seed] if item["train_percentage"] == percentage))["test_f1"] * 100
        for percentage in percentages
    ]
    return test_f1s

def extract_accuracies_sens(data, seed):
    percentages = [number_560, number_320, number_160, number_80, number_40, number_20, number_10, number_5]
    test_sens = [
        (next(item for item in data[seed] if item["train_percentage"] == percentage))["test_sens"] * 100
        for percentage in percentages
    ]
    return test_sens


def extract_accuracies_spec(data, seed):
    percentages = [number_560, number_320, number_160, number_80, number_40, number_20, number_10, number_5]
    test_spec = [
        (next(item for item in data[seed] if item["train_percentage"] == percentage))["test_spec"] * 100
        for percentage in percentages
    ]
    return test_spec
