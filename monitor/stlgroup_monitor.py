import numpy as np
from scipy import stats
import pandas as pd

import config

stl_filename_1="/rob_save21-run_time[1].csv"
stl_filename_2="/rob_save22-run_time[1].csv"


def get_data(scenario_type,file_code):
    global stl_filename_1
    global stl_filename_2
    # result_path = "./rob_result/crossroad/" + config.ScenarioGeneratorConfig.file_vinit
    result_path = "./rob_result/"+scenario_type+"/" + file_code

    result1 = pd.read_csv(result_path + stl_filename_1)
    result2 = pd.read_csv(result_path + stl_filename_2)

    # 推荐使用的方法
    array_2d = np.column_stack((result1.iloc[:, 0], result2.iloc[:, 0]))

    return array_2d


def robust_combined_measure(scenario_type,file_code):
    """
    组合输出策略：根据数据符号特性映射到三个不重叠范围，同时反映数据分布

    参数:
        data: 数值列表或数组

    返回:
        float: 映射后的结果值
    """
    data_all=get_data(scenario_type,file_code)
    output_list=[]

    for data_raw in data_all:
        data = np.array(data_raw)
        n = len(data)

        # 1. 符号特性判断
        all_negative = np.all(data < 0)
        all_positive = np.all(data > 0)
        mixed_signs = not (all_negative or all_positive)

        # 2. 计算数据的分布特征（使用鲁棒性方法）
        if n == 1:
            # 单一数据点
            base_value = data[0]
        elif n == 2:
            # 两个数据点使用算术平均
            base_value = np.mean(data)
        else:
            # 三个及以上数据点使用修整平均（20%修整）
            base_value = stats.trim_mean(data, 0.2)

        # 3. 计算数据的离散程度（使用鲁棒性方法）
        if n <= 2:
            spread = 0.1  # 小样本使用固定离散度
        else:
            # 使用四分位距（IQR）作为离散度度量
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            spread = iqr / (np.median(np.abs(data)) + 1e-8)  # 相对离散度

        # 4. 根据符号特性进行范围映射
        if all_negative:
            # 全部小于0：映射到 (-∞, -1)
            # 使用负数的幂平均增强区分度
            negative_data = np.abs(data)  # 取绝对值处理
            p_mean = np.power(np.mean(np.power(negative_data, 0.7)), 1 / 0.7)

            # 基础映射：-2到-10范围
            base_output = -1 - 9 * (1 - 1 / (1 + p_mean))

            # 加入离散度调整
            output = base_output - spread

        elif all_positive:
            # 全部大于0：映射到 (1, +∞)
            # 使用幂平均增强大值的权重
            p_mean = np.power(np.mean(np.power(data, 1.3)), 1 / 1.3)

            # 基础映射：1到10范围
            base_output = 1 + 9 * (1 - 1 / (1 + p_mean))

            # 加入离散度调整
            output = base_output + spread

        else:
            # 部分小于0：映射到 [-1, 1]
            # 使用中位数作为中心趋势的鲁棒估计
            # median_val = np.mean(data)
            #
            # # 计算数据的偏斜程度
            # negative_ratio = np.sum(data < 0) / n
            # positive_ratio = np.sum(data > 0) / n

            shifted_p_mean = np.power(np.mean(np.power(data+10, 0.7)), 1 / 0.7)
            p_mean=shifted_p_mean-10
            output = np.tanh(p_mean)
            # # 基础映射：基于中位数和正负比例
            # if median_val == 0:
            #     base_output = 0
            # else:
            #     # 使用双曲正切函数将中位数压缩到[-1,1]
            #     # base_output = np.tanh(median_val / (np.std(data) + 1e-8))
            #     output = np.tanh(median_val )

            # 根据正负比例进行微调
            # balance_adjust = (positive_ratio - negative_ratio) * 0.3
            # output = np.clip(base_output + balance_adjust, -1, 1)

        output_list.append(output)
        # 创建DataFrame，指定列名
        df = pd.DataFrame(output_list, columns=['column_name'])

        # 保存到CSV文件
        output_path="./monitor/output_"+config.ScenarioGeneratorConfig.file_vinit+".csv"
        df.to_csv(output_path, index=False)


