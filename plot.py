import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import config

def plot(scenario_type):
    path = './rob_result/'+scenario_type+"/"+config.ScenarioGeneratorConfig.file_vinit

    os.makedirs(path+"/figs", exist_ok=True)
    path_list = os.listdir(path)


    #print(path_list)

    plt.figure(figsize=(12, 6))  # 设置画布大小
    for filename in path_list:
        pre_filename=filename.split("-")
        if filename != "figs" and filename != "raw" and pre_filename[0] != "rawdata":
            # f = open(os.path.join(path, filename), 'rb')
            data = pd.read_csv(os.path.join(path,filename), encoding='ISO-8859-1')
            total_rows = data.shape[0]
            # csv_path="./rob_result/"+scenario_type+"/rob_save"+index+".csv"

            x=np.arange(0,total_rows,1)
            plt.plot(x, data['rob'],
                     color='#FF6B6B',  # 十六进制颜色码
                     linestyle='--',
                     linewidth=2,
                     marker='o')

            plt.title('stl1', fontsize=14, fontweight='bold')
            plt.xlabel('time_step', fontsize=12)
            plt.ylabel('rob', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.7)  # 半透明网格线
            # plt.xticks(rotation=45)  # X轴标签旋转45度
            plt.tight_layout()  # 自动调整布局

            name=filename.split('.')[0]
            plt_path = path + "/figs/"+ name + ".png"
            plt.savefig(plt_path)
            plt.clf()
    plt.close("all")
            # plt.show()


if __name__ == '__main__':
    plot("change_lane")