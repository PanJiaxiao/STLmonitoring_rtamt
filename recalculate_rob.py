import sys
import rtamt
import csv
import os
import argparse
from plot import plot
from utils import clear_result

import numpy as np


def monitor(data, filename, index: str, phi_str: str, signal_str, scenario_type, run_time):
    # # stl
    #print("arrive")
    name = []
    value = []
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var('a', 'float')
    spec.declare_var('b', 'float')
    spec.spec = phi_str
    for i, signal in enumerate(signal_str):
        #print(signal_str)
        #print(data)
        if signal == "a_bef_b":
            spec.declare_var(signal, 'int')
        else:
            spec.declare_var(signal, 'float')

        name.append(signal)
        value.append(data[i])
    data_input = zip(name, value)

    try:
        spec.parse()
        spec.pastify()
    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()

    rob = spec.update(0, list(data_input))
    #print('time=' + str(0) + ' rob=' + str(rob))

    #write
    filepath = "./rob_result/" + "recalculate" + "/"  +scenario_type+ str(index) + "-run_time" + str(run_time) + ".csv"

    if os.path.exists(filepath):
        data_write = [[str(rob)]]
        with open(filepath, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_write)

    else:
        data_write = [["rob"]]
        with open(filepath, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_write)



def pre_monitor( scenario_type, run_time):
    if scenario_type == "change_lane":
        stl_dict = {
            "stl_1": {"phi_str": "lan_o_a>=0.01",
                      "signal_str": "lan_o_a"},
            "stl_2": {"phi_str": "lan_o_a<0.01",
                      "signal_str": "lan_o_a"},
            "stl_3": {"phi_str": "va_acc>=0.01",
                      "signal_str": "va_acc"},
            "stl_4": {"phi_str": "va_acc<0.01",
                      "signal_str": "va_acc"},
            "stl_5": {"phi_str": "(va_acc>=0.01) and (lan_o_a<0.01)",
                      "signal_str": "va_acc,lan_o_a"},
            "stl_6": {"phi_str": "(va_acc<0.01) and (lan_o_a<0.01)",
                      "signal_str": "va_acc,lan_o_a"},
            "stl_7": {"phi_str": "(va_acc>=0.01) and (lan_o_a>=0.01)",
                      "signal_str": "va_acc,lan_o_a"},
            "stl_8": {"phi_str": "(va_acc<0.01) and (lan_o_a>=0.01)",
                      "signal_str": "va_acc,lan_o_a"},
            "stl_9": {"phi_str": "(lan_o_b<0.01) and (distance<40) and (va_acc>=0.01) and (lan_o_a<0.01)",
                      "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_10": {"phi_str": "(lan_o_a<0.01) and (distance<40) and (vb_acc<0.01) and (lan_o_b<0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_11": {"phi_str": "(lan_o_b<0.01) and (distance<40) and (va_acc>=0.01) and (lan_o_a<0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_12": {"phi_str": "(lan_o_a<0.01) and (distance<40) and (vb_acc>=0.01) and (lan_o_b<0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_13": {"phi_str": "(lan_o_b<0.01) and (distance<40) and (va_acc<0.01) and (lan_o_a<0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_14": {"phi_str": "(lan_o_a<0.01) and (distance<40) and (vb_acc>=0.01) and (lan_o_b<0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_15": {"phi_str": "(lan_o_b<0.01) and (distance<40) and (va_acc<0.01) and (lan_o_a<0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_16": {"phi_str": "(lan_o_a<0.01) and (distance<40) and (vb_acc<0.01) and (lan_o_b<0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_17": {"phi_str": "(lan_o_b<0.01) and (distance<40) and (va_acc>=0.01) and (lan_o_a>=0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_18": {"phi_str": "(lan_o_a>=0.01) and (distance<40) and (vb_acc<0.01) and (lan_o_b<0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_19": {"phi_str": "(lan_o_b<0.01) and (distance<40) and (va_acc<0.01) and (lan_o_a<0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_20": {"phi_str": "(lan_o_a<0.01) and (distance<40) and (vb_acc<0.01) and (lan_o_b<0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_21": {"phi_str": "(lan_o_b<0.01) and (distance<40) and (va_acc<0.01) and (lan_o_a>=0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_22": {"phi_str": "(lan_o_a>=0.01) and (distance<40) and (vb_acc>0.01) and (lan_o_b<0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_23": {"phi_str": "(lan_o_b<0.01) and (distance<40) and (va_acc<0.01) and (lan_o_a>=0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_24": {"phi_str": "(lan_o_a>=0.01) and (distance<40) and (vb_acc<=0.01) and (lan_o_b<0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_25": {"phi_str": "(lan_o_b<0.01) and (distance<40) and (va_acc<0.01) and (lan_o_a>=0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_26": {"phi_str": "(lan_o_a>=0.01) and (distance<40) and (vb_acc>=0.01) and (lan_o_b<0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_27": {"phi_str": "(lan_o_b>=0.01) and (distance<40) and (va_acc>0.01) and (lan_o_a<0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_28": {"phi_str": "(lan_o_a<0.01) and (distance<40) and (vb_acc<0.01) and (lan_o_b>=0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_29": {"phi_str": "(lan_o_b>=0.01) and (distance<40) and (va_acc>=0.01) and (lan_o_a<0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_30": {"phi_str": "(lan_o_a<0.01) and (distance<40) and (vb_acc>=0.01) and (lan_o_b>=0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_31": {"phi_str": "(lan_o_b>=0.01) and (distance<40) and (va_acc<0.01) and (lan_o_a<0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_32": {"phi_str": "(lan_o_a<0.01) and (distance<40) and (vb_acc>0.01) and (lan_o_b>=0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_33": {"phi_str": "(lan_o_b>=0.01) and (distance<40) and (va_acc<0.01) and (lan_o_a<0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_34": {"phi_str": "(lan_o_a<0.01) and (distance<40) and (vb_acc<0.01) and (lan_o_b>=0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_35": {"phi_str": "(lan_o_b>=0.01) and (distance<40) and (va_acc>=0.01) and (lan_o_a>=0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_36": {"phi_str": "(lan_o_a>=0.01) and (distance<40) and (vb_acc<0.01) and (lan_o_b>=0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_37": {"phi_str": "(lan_o_b>=0.01) and (distance<40) and (va_acc>=0.01) and (lan_o_a>=0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_38": {"phi_str": "(lan_o_a>=0.01) and (distance<40) and (vb_acc>=0.01) and (lan_o_b>=0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_39": {"phi_str": "(lan_o_b>=0.01) and (distance<40) and (va_acc<0.01) and (lan_o_a>=0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_40": {"phi_str": "(lan_o_a>=0.01) and (distance<40) and (vb_acc>=0.01) and (lan_o_b>=0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},
            "stl_41": {"phi_str": "(lan_o_b>=0.01) and (distance<40) and (va_acc>=0.01) and (lan_o_a>=0.01)",
                       "signal_str": "lan_o_b,distance,va_acc,lan_o_a"},
            "stl_42": {"phi_str": "(lan_o_a>=0.01) and (distance<40) and (vb_acc<0.01) and (lan_o_b>=0.01)",
                       "signal_str": "lan_o_a,distance,vb_acc,lan_o_b"},

        }
    elif scenario_type == "crossroad":
        # not add forward and backward constrane
        stl_dict = {
            # basic movenment
            "stl_1": {"phi_str": "(va_acc >= 0.0001) and (abs(yr_a) >= 0.0001)",
                      "signal_str": "va_acc,yr_a"},
            "stl_2": {"phi_str": "(vb_acc >= 0.0001) and (abs(yr_b) >= 0.0001)",
                      "signal_str": "vb_acc,yr_b"},
            "stl_3": {"phi_str": "(va_acc < 0.0001) and (abs(yr_a) >= 0.0001)",
                      "signal_str": "va_acc,yr_a"},
            "stl_4": {"phi_str": "(vb_acc < 0.0001) and (abs(yr_b) >= 0.0001)",
                      "signal_str": "vb_acc,yr_b"},
            "stl_5": {"phi_str": "(va_acc >= 0.0001) and (abs(yr_a) < 0.0001)",
                      "signal_str": "va_acc,yr_a"},
            "stl_6": {"phi_str": "(vb_acc >= 0.0001) and (abs(yr_b) < 0.0001)",
                      "signal_str": "vb_acc,yr_b"},
            "stl_7": {"phi_str": "(va_acc < 0.0001) and (abs(yr_a) < 0.0001)",
                      "signal_str": "va_acc,yr_a"},
            "stl_8": {"phi_str": "(vb_acc < 0.0001) and (abs(yr_b) < 0.0001)",
                      "signal_str": "vb_acc,yr_b"},
            # basic interaction
            "stl_9": {
                "phi_str": "((va_acc >= 0.0001) and (abs(yr_a) >= 0.001)) and ((vb_acc >= 0.0001) and (abs(yr_b) >= 0.0001))",
                "signal_str": "va_acc,yr_a,vb_acc,yr_b"},
            "stl_10": {
                "phi_str": "((va_acc >= 0.0001) and (abs(yr_a) >= 0.001)) and ((vb_acc < 0.0001) and (abs(yr_b) >= 0.0001)) or ((vb_acc >= 0.0001) and (abs(yr_b) >= 0.0001)) and ((va_acc < 0.0001) and (abs(yr_a) >= 0.0001))",
                "signal_str": "vb_acc,yr_b,va_acc,yr_a"},
            "stl_11": {
                "phi_str": "((va_acc >= 0.0001) and (abs(yr_a) >= 0.001) and (vb_acc >= 0.0001) and (abs(yr_b) < 0.0001)) or ((vb_acc < 0.0001) and (abs(yr_b) >= 0.0001) and (va_acc >= 0.0001) and (abs(yr_a) < 0.0001))",
                "signal_str": "va_acc,yr_a,vb_acc,yr_b"},
            "stl_12": {
                "phi_str": "((va_acc >= 0.0001) and (abs(yr_a) >= 0.001) and (vb_acc < 0.0001) and (abs(yr_b) < 0.0001)) or ((vb_acc >= 0.0001) and (abs(yr_b) >= 0.0001) and (va_acc < 0.0001) and (abs(yr_a) < 0.0001))",
                "signal_str": "va_acc,yr_a,vb_acc,yr_b"},
            "stl_13": {
                "phi_str": "(va_acc < 0.0001) and (abs(yr_a) >= 0.001) and ((vb_acc < 0.0001) and (abs(yr_b) >= 0.0001))",
                "signal_str": "va_acc,yr_a,vb_acc,yr_b"},
            "stl_14": {
                "phi_str": "((va_acc < 0.0001) and (abs(yr_a) >= 0.001) and (vb_acc >= 0.0001) and (abs(yr_b) < 0.0001)) or (vb_acc < 0.0001) and (abs(yr_b) >= 0.0001) and (va_acc >= 0.0001) and (abs(yr_a) < 0.0001)",
                "signal_str": "va_acc,yr_a,vb_acc,yr_b"},
            "stl_15": {
                "phi_str": "((va_acc < 0.0001) and (abs(yr_a) >= 0.001) and (vb_acc < 0.0001) and (abs(yr_b) < 0.0001)) or ((vb_acc < 0.0001) and (abs(yr_b) >= 0.0001) and (va_acc < 0.0001) and (abs(yr_a) < 0.0001))",
                "signal_str": "va_acc,yr_a,vb_acc,yr_b"},
            "stl_16": {
                "phi_str": "(va_acc >= 0.0001) and (abs(yr_a) < 0.001) and (vb_acc >= 0.0001) and (abs(yr_b) < 0.0001)",
                "signal_str": "va_acc,yr_a,vb_acc,yr_b"},
            "stl_17": {
                "phi_str": "((va_acc >= 0.0001) and (abs(yr_a) < 0.001) and (vb_acc < 0.0001) and (abs(yr_b) < 0.0001)) or ((vb_acc >= 0.0001) and (abs(yr_b) < 0.0001) and (va_acc < 0.0001) and (abs(yr_a) < 0.0001))",
                "signal_str": "va_acc,yr_a,vb_acc,yr_b"},
            "stl_18": {"phi_str": "(va_acc < 0.0001) and (abs(yr_a) < 0.001) and (vb_acc < 0.0001) and (abs(yr_b) < 0.0001)",
                       "signal_str": "va_acc,yr_a,vb_acc,yr_b"},
            # interaction under scenrio
            "stl_19": {
                "phi_str": "(yr_b*yr_a<0) and (abs(yr_b) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (va_acc >= 0.0001) and (abs(yr_a) >= 0.0001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_20": {
                "phi_str": "(yr_b*yr_a<0) and (abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (vb_acc >= 0.0001) and (abs(yr_b) >= 0.0001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_21": {
                "phi_str": "(yr_b*yr_a<0) and (abs(yr_b) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (va_acc < 0.0001) and (abs(yr_a) >= 0.0001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_22": {
                "phi_str": "(yr_b*yr_a<0) and (abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (vb_acc >= 0.0001) and (abs(yr_b) >= 0.0001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_23": {
                "phi_str": "(yr_b*yr_a<0) and (abs(yr_b) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (va_acc < 0.0001) and (abs(yr_a) >= 0.0001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_24": {
                "phi_str": "(yr_b*yr_a<0) and (abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (vb_acc < 0.0001) and (abs(yr_b) >= 0.0001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_25": {
                "phi_str": "(abs(yr_b) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (va_acc >= 0.0001) and (abs(yr_a) >= 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_26": {
                "phi_str": "(abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (vb_acc >= 0.0001) and (abs(yr_b) < 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_27": {
                "phi_str": "(abs(yr_b) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (va_acc >= 0.0001) and (abs(yr_a) >= 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_28": {
                "phi_str": "(abs(yr_b) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (vb_acc < 0.0001) and (abs(yr_b) < 0.001)",
                "signal_str": "yr_b,distance,yr_a,vb_acc,orientation_b,orientation_a"},
            "stl_29": {
                "phi_str": "(abs(yr_b) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (va_acc < 0.0001) and (abs(yr_a) >= 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_30": {
                "phi_str": "(abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (vb_acc >= 0.0001) and (abs(yr_b) < 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_31": {
                "phi_str": "(abs(yr_b) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (va_acc < 0.0001) and (abs(yr_a) >= 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_32": {
                "phi_str": "(abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)<1.572) and (vb_acc < 0.0001) and (abs(yr_b) < 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_33": {
                "phi_str": "(yr_a*yr_b>0) and (abs(yr_b) >= 0.001)and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (va_acc >= 0.0001) and (abs(yr_a) >= 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_34": {
                "phi_str": "(yr_a*yr_b>0) and (abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (vb_acc >= 0.0001) and (abs(yr_b) >= 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_35": {
                "phi_str": "(yr_a*yr_b>0) and (abs(yr_b) >= 0.001)and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (va_acc < 0.0001) and (abs(yr_a) >= 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_36": {
                "phi_str": "(yr_a*yr_b>0) and (abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (vb_acc >= 0.0001) and (abs(yr_b) >= 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_37": {
                "phi_str": "(yr_a*yr_b>0) and (abs(yr_b) >= 0.001)and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (va_acc < 0.0001) and (abs(yr_a) >= 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_38": {
                "phi_str": "(yr_a*yr_b>0) and (abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (vb_acc < 0.0001) and (abs(yr_b) >= 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_39": {
                "phi_str": "(abs(yr_b) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (abs(orientation_a-orientation_b)<=3.142) and (va_acc >= 0.0001) and (abs(yr_a) >= 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_40": {
                "phi_str": "(abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (abs(orientation_a-orientation_b)<=3.142) and (vb_acc >= 0.0001) and (abs(yr_b) < 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_41": {
                "phi_str": "(abs(yr_b) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (abs(orientation_a-orientation_b)<=3.142) and (va_acc < 0.0001) and (abs(yr_a) >= 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_42": {
                "phi_str": "(abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (abs(orientation_a-orientation_b)<=3.142) and (vb_acc >= 0.0001) and (abs(yr_b) < 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_43": {
                "phi_str": "(abs(yr_b) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (abs(orientation_a-orientation_b)<=3.142) and (va_acc < 0.0001) and (abs(yr_a) >= 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_44": {
                "phi_str": "(abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (abs(orientation_a-orientation_b)<=3.142) and (vb_acc < 0.0001) and (abs(yr_b) < 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_45": {
                "phi_str": "(abs(yr_b) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (va_acc >= 0.0001) and (abs(yr_a) < 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_46": {
                "phi_str": "(abs(yr_a) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (vb_acc >= 0.0001) and (abs(yr_b) < 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_47": {
                "phi_str": "(abs(yr_b) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (va_acc < 0.0001) and (abs(yr_a) < 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_48": {
                "phi_str": "(abs(yr_a) >= 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (vb_acc >= 0.0001) and (abs(yr_b) >= 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},
            "stl_49": {
                "phi_str": "(abs(yr_b) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (va_acc < 0.0001) and (abs(yr_a) < 0.001)",
                "signal_str": "yr_b,distance,yr_a,va_acc,orientation_b,orientation_a"},
            "stl_50": {
                "phi_str": "(abs(yr_a) < 0.001) and (distance<40) and (abs(orientation_a-orientation_b)>=1.572) and (vb_acc < 0.0001) and (abs(yr_b) < 0.001)",
                "signal_str": "yr_a,distance,yr_b,vb_acc,orientation_b,orientation_a"},

        }
    elif scenario_type == "single_straight":
        flage = 0
        # TODO put "a_bef_b" first
        stl_dict = {
            "stl_1": {"phi_str": "va_acc>0.0001",
                      "signal_str": "va_acc"},
            "stl_2": {"phi_str": "va_acc<-0.0001",
                      "signal_str": "va_acc"},
            "stl_3": {"phi_str": "abs(va_acc)<=0.0001",
                      "signal_str": "va_acc"},
            "stl_4": {"phi_str": "(va_acc>0.0001) and (vb_acc>0.0001)",
                      "signal_str": "va_acc,vb_acc"},
            "stl_5": {"phi_str": "(va_acc>0.0001) and (vb_acc<-0.0001)",
                      "signal_str": "va_acc,vb_acc"},
            "stl_6": {"phi_str": "(va_acc>0.0001) and (abs(vb_acc)<=0.0001)",
                      "signal_str": "va_acc,vb_acc"},
            "stl_7": {"phi_str": "(va_acc<-0.0001) and (vb_acc<-0.0001)",
                      "signal_str": "va_acc,vb_acc"},
            "stl_8": {"phi_str": "(va_acc>0.0001) and (vb_acc<-0.0001)",
                      "signal_str": "va_acc,vb_acc"},
            "stl_9": {"phi_str": "(abs(va_acc)<=0.0001) and (abs(vb_acc)<=0.0001)",
                      "signal_str": "va_acc,vb_acc"},
            "stl_10": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (va_acc>0.0001)) or ((distance<20) and (a_bef_b<0) and (vb_acc>0.0001)) ",
                "signal_str": "a_bef_b,distance,va_acc,vb_acc"},
            "stl_11": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (vb_acc>0.0001) ) or ((distance<20) and (a_bef_b<0) and (va_acc>0.0001))",
                "signal_str": "a_bef_b,distance,va_acc,vb_acc"},
            "stl_12": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (va_acc>0.0001) ) or ((distance<20) and (a_bef_b<0) and (vb_acc>0.0001))",
                "signal_str": "a_bef_b,distance,va_acc,vb_acc"},
            "stl_13": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (vb_acc<-0.0001)) or ((distance<20) and (a_bef_b<0) and (va_acc<-0.0001))",
                "signal_str": "a_bef_b,distance,vb_acc,va_acc"},
            "stl_14": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (va_acc>0.0001) ) or ((distance<20) and (a_bef_b<0) and (va_acc>0.0001))",
                "signal_str": "a_bef_b,distance,va_acc,vb_acc"},
            "stl_15": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (abs(vb_acc)<=0.0001)) or ((distance<20) and (a_bef_b<0) and (abs(va_acc)<=0.0001)) ",
                "signal_str": "a_bef_b,distance,vb_acc,va_acc"},
            "stl_16": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (va_acc<-0.0001)) or ((distance<20) and (a_bef_b<0) and (vb_acc<-0.0001)) ",
                "signal_str": "a_bef_b,distance,va_acc,vb_acc"},
            "stl_17": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (vb_acc>0.0001)) or ((distance<20) and (a_bef_b<0) and (va_acc>0.0001)) ",
                "signal_str": "a_bef_b,distance,vb_acc,va_acc"},
            "stl_18": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (va_acc<-0.0001)) or ((distance<20) and (a_bef_b<0) and (vb_acc<-0.0001)) ",
                "signal_str": "a_bef_b,distance,va_acc,vb_acc"},
            "stl_19": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (vb_acc<-0.0001)) or ((distance<20) and (a_bef_b<0) and (va_acc<-0.0001)) ",
                "signal_str": "a_bef_b,distance,vb_acc,va_acc"},
            "stl_20": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (va_acc<-0.0001)) or ((distance<20) and (a_bef_b<0) and (vb_acc<-0.0001)) ",
                "signal_str": "a_bef_b,distance,va_acc,vb_acc"},
            "stl_21": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (abs(vb_acc)<=0.0001) ) or ((distance<20) and (a_bef_b<0) and (abs(va_acc)<=0.0001))",
                "signal_str": "a_bef_b,distance,vb_acc,va_acc"},
            "stl_22": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (abs(va_acc)<=0.0001)) or ((distance<20) and (a_bef_b<0) and (abs(vb_acc)<=0.0001)) ",
                "signal_str": "a_bef_b,distance,va_acc,vb_acc"},
            "stl_23": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (vb_acc>0.0001)) or ((distance<20) and (a_bef_b<0) and (va_acc>0.0001)) ",
                "signal_str": "a_bef_b,distance,vb_acc,va_acc"},
            "stl_24": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (abs(va_acc)<=0.0001)) or ((distance<20) and (a_bef_b<0) and (abs(vb_acc)<=0.0001)) ",
                "signal_str": "a_bef_b,distance,va_acc,vb_acc"},
            "stl_25": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (vb_acc<-0.0001)) or ((distance<20) and (a_bef_b<0) and (va_acc<-0.0001)) ",
                "signal_str": "a_bef_b,distance,vb_acc,va_acc"},
            "stl_26": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (abs(va_acc)<=0.0001)) or ((distance<20) and (a_bef_b<0) and (abs(vb_acc)<=0.0001)) ",
                "signal_str": "a_bef_b,distance,va_acc,vb_acc"},
            "stl_27": {
                "phi_str": "((distance<20) and (a_bef_b>0) and (abs(vb_acc)<=0.0001)) or ((distance<20) and (a_bef_b<0) and (abs(va_acc)<=0.0001)) ",
                "signal_str": "a_bef_b,distance,vb_acc,va_acc"}

        }
    # 先存储所有基础数据


    for index, stl_index in enumerate(stl_dict, start=1):
        #TODO get data from csv
        phi_str = stl_dict[stl_index]["phi_str"]
        signal_str = stl_dict[stl_index]["signal_str"]

        # print(signal_str)
        signal_str_list = signal_str.split(",")
        # print(signal_str_list)

        """
        读取CSV文件，每行数据作为一个列表返回
        """
        filename = "./rob_result/" + scenario_type + "/raw/" + "rob_save"+str(index) +"-run_time"+ str(run_time) + ".csv"
        data_result = []
        with open(filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data_result.append(row)
                new_row=[]
                f=0
                for i in row:
                    try:
                        new_row.append(int(i))
                        #print("arrive int")
                        #print(row)
                    except ValueError:
                        # 再尝试浮点数转换
                        try:
                            new_row.append(float(i))
                            #print("arrive float")
                            #print(row)
                        except ValueError:
                            #print("arrive str")
                            #print(row)
                            f=1
                if f==0:
                    monitor(new_row, filename, str(index), phi_str, signal_str_list, scenario_type, run_time)
            plot("recalculate")




        # TODO monitor complete



if __name__ == '__main__':
    parser = argparse.ArgumentParser("run_multi_motion_planner_system")
    parser.add_argument("--scenario_type", type=str,
                        choices=["change_lane", "crossroad", "single_straight"],
                        default="change_lane", help="Select a scenarios type for simulations")
    parser.add_argument("--run_time", type=int, nargs='*', default=0,
                        help="run time.")
    args = parser.parse_args()
    clear_result("./rob_result/" + "recalculate")
    pre_monitor(args.scenario_type,args.run_time)

