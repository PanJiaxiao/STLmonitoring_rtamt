import sys
import rtamt
import csv
import os
import numpy as np

import config

#monitor(data_result, filename, str(index), phi_str, signal_str_list, scenario_type, run_time)
def monitor(data, filename ,index:str, phi_str:str,signal_str,scenario_type,run_time):
    # # stl
    name=[]
    value=[]
    spec = rtamt.StlDiscreteTimeSpecification()
    spec.declare_var('a', 'float')
    spec.declare_var('b', 'float')
    spec.spec = phi_str
    for i,signal in enumerate(signal_str):
        if signal=="a_bef_b":
            spec.declare_var(signal,'int')
        else:
            spec.declare_var(signal,'float')

        name.append(signal)
        value.append(data[i])
    data_input=zip(name,value)

    try:
        spec.parse()
        spec.pastify()
    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()

    rob = spec.update(0, list(data_input))
    #print('time=' + str(0) + ' rob=' + str(rob))
    return rob


    # filepath="./rob_result/"+scenario_type+"/"+config.ScenarioGeneratorConfig.file_vinit+"/"+filename+str(index)+"-run_time"+str(run_time)+".csv"
    # filepath_2 = "./rob_result/" + scenario_type+"/" +config.ScenarioGeneratorConfig.file_vinit+"/raw/" + filename + str(index) + "-run_time" + str(run_time) + ".csv"
    # os.makedirs("./rob_result/" + scenario_type +"/"+config.ScenarioGeneratorConfig.file_vinit+"/"+ "/raw", exist_ok=True)
    #
    # if os.path.exists(filepath) :
    #     data_write=[[str(rob)]]
    #     with open(filepath, mode='a', encoding='utf-8', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerows(data_write)
    #
    #     with open(filepath_2, mode='a', encoding='utf-8', newline='') as file:
    #         writer = csv.writer(file)
    #         data_converted = [item.item() if hasattr(item, 'item') else item for item in value]
    #         writer.writerows([data_converted])
    # else:
    #     data_write=[["rob"]]
    #     with open(filepath, mode='w', encoding='utf-8', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerows(data_write)
    #     with open(filepath_2, mode='w', encoding='utf-8', newline='') as file:
    #         writer = csv.writer(file)
    #         data_converted = [item.item() if hasattr(item, 'item') else item for item in name]
    #         writer.writerows([data_converted])


if __name__ == '__main__':

    monitor()