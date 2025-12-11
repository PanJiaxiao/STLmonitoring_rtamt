import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import subprocess
import random
import argparse
import traceback

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.termination.default import MultiObjectiveDefaultTermination


import config
from run_multi_motion_planner_system import get_init_scenario, ScenarioGenerator, simulate, get_random_scenario, \
    get_mutated_scenario
import utils
from plotter.plotter import save_images_as_gif, scenario_to_image
from commonroad.scenario.lanelet import LaneletNetwork
from utils import SimulationResult, PlanningStatus, create_cc_scenario, get_cmap, prepare_scenario, write_scenario, \
    save_for_stl, clear_result
from plot import plot
from monitor.stlgroup_monitor import robust_combined_measure

stl_filename_1="/rob_save24-run_time[1].csv"
stl_filename_2="/rob_save25-run_time[1].csv"



def run_commonroad():
    global filename
    global scenario_type

    result_init = get_init_scenario(args.scenario_type,  filename)
    if result_init is False:
        return False
    best_scenario, best_problem_set = result_init

    #mark1
    scenario_generator = ScenarioGenerator(best_scenario)
    max_steps = scenario_generator.config.max_time_step
    result_name = f"{filename.split('.')[0]}_{args.criticality_mode}_{args.scenario_type}_{args.run_time}_{config.ScenarioGeneratorConfig.file_vinit}"

    # simulate base scenario to find starting values
    best_result = simulate(best_scenario, best_problem_set, args.scenario_type, args.run_time, max_steps=max_steps)
    max_avg_danger, max_danger, max_interactions, max_avg_pp_id, max_danger_pp_id = utils.compute_criticality_metrics(
        best_result)
    best_criticality_pp_id, best_criticality = utils.get_criticality_metric(avg_danger=max_avg_danger,
                                                                            max_danger=max_danger,
                                                                            max_interactions=max_interactions,
                                                                            max_avg_pp_id=max_avg_pp_id,
                                                                            max_danger_pp_id=max_danger_pp_id,
                                                                            mode=args.criticality_mode)

    best_result.overall_critically=best_criticality
    best_result.simulation_number=0
    # save gif of initial simulation
    sim_filename = filename.split('.')[0] + "_sim_0"
    save_images_as_gif(best_result.images, result_name, sim_filename)
    result_iteration = 0
    # initial simulation is stored at index 0
    results = [best_result]

    utils.save_as_csv(results, filename, result_name)
    # plot(args.scenario_type)

    lanelet_network: LaneletNetwork = best_scenario.lanelet_network

    # get most critical planning problem that remains in the final test scenario
    # critical_key = best_criticality_pp_id
    # print(f"Most critical key - criticality: {critical_key} - {best_criticality}, sim_nr: {result_iteration}")

    xml_path = write_scenario(filename, best_result.simulation_state_map, best_result.scenario, best_result.problem_set,
                              result_name)

    save_for_stl(filename, args.scenario_type, args.run_time, xml_path, lanelet_network)
    robust_combined_measure(args.scenario_type,config.ScenarioGeneratorConfig.file_vinit)

    if args.run_result:
        pure_filename = filename.split('.')[0]
        planner_path = os.path.abspath(os.getcwd() + "/planners/reactive_planner_zipped_latest/run_combined_planner.py")
        subprocess.run(
            " ".join(
                ["python",
                 planner_path,
                 "--base_dir",
                 f"./plots/{pure_filename}", "--filename", pure_filename + "-multi-planner-simulated.xml"]), shell=True)

    return True


def function_a(parameters):
    global stl_filename_1
    global stl_filename_2
    global scenario_type
    """
    模拟函数a，接收参数，返回两个随时间变化的结果
    """
    vinit1, start1,  acc1 ,vinit2, start2, acc2 = parameters

    config.ScenarioGeneratorConfig.file_vinit = str(vinit1+start1+acc1+vinit2+start2+acc2)
    print("ga_id:",config.ScenarioGeneratorConfig.file_vinit)

    print("para@@@@@@", parameters)

    vinit = [vinit1, vinit2]
    start = [start1, start2]

    acc=[acc1,acc2]

    config.ScenarioGeneratorConfig.vinit = vinit
    config.ScenarioGeneratorConfig.acc = acc
    config.ScenarioGeneratorConfig.start = start

    if run_commonroad() == False:
        return False
#TODO
    result_path = "./monitor/output_"+config.ScenarioGeneratorConfig.file_vinit+".csv"
    result = pd.read_csv(result_path)

    return np.array(result)


class MultiObjectiveProblem(ElementwiseProblem):
    """
    使用pymoo定义多目标优化问题
    """

    def __init__(self):
        # 定义6个参数的范围
        #vinit1, start1,  acc1 ,vinit2, start2, acc2
        xl = [2.0, 0.0,  0.0,2.0, 0.0, 0.0 ] # 下界
        xu = [8.0, 1.0, 1.0,8.0, 1.0, 1.0 ] # 上界

        # 2个目标（都是最大化）
        super().__init__(n_var=6, n_obj=1, xl=xl, xu=xu)


    def _evaluate(self, x, out, *args, **kwargs):
        """
        评估函数：计算两个目标
        目标1：两个结果都大于0的时间点数量（最大化）
        目标2：两个结果大于0的程度的平均值（最大化）
        """
        """
                评估函数2：计算两个目标
                目标1：stl1大于0的时间点数量（最大化）
                目标2：stl2大于0的时间点数量（最大化）
        """
        try:
            result = function_a(x)
            if result is False:
                # 如果function_a返回False，直接给予惩罚
                #function1
                out["F"] = [10]  # 严重惩罚
                #function2
                # out["F"] = [10, 10]
                #fucntion4
                # out["F"] = [0, 0, 0]
                return
        except (IndexError, TypeError) as e:
            # 捕获IndexError，直接给予惩罚
            print(f"索引错误: {e}")
            traceback.print_exc()
            #function1
            out["F"] = [10]  # 严重惩罚
            #function2
            # out["F"] = [10, 10]
            return

        #function1
        # 计算两个结果都大于0的时间点
        both_positive = (result > 1)
        positive_count = np.sum(both_positive)

        part_positive =(result > -1)
        part_count = np.sum(part_positive)


        #function1
        # 如果没有同时大于0的时刻，给予惩罚
        if positive_count == 0:
            out["F"] = [10]  # 严重惩罚

            return


        # 计算大于-1的程度（只在两个都大于0时计算）
        min_positive_values = result[part_positive]
        positive_magnitude = np.mean(min_positive_values)


        #function1
        # 返回两个目标值（注意：pymoo默认是最小化，所以这里取负号来转换为最大化）
        if positive_magnitude >=1:
            positive_magnitude = positive_magnitude+ 2.0
            part_count = part_count + 2.0

        elif positive_magnitude >= -1.0:
            positive_magnitude = positive_magnitude
            part_count = part_count - 1.0
        else:
            positive_magnitude =positive_magnitude-10.0
            part_count = part_count - 10.0

        out["F"] = [-part_count]
        # out["F"] = [-positive_count, -positive_magnitude]
        #function2
        # out["F"] = [-positive_count_1, -positive_count_2]
        #function3
        # out["F"] = [-positive_magnitude_1, -positive_magnitude_2]



def run_pymoo_optimization(pop_size=2, generations=5):
    """使用pymoo运行多目标优化"""

    # 创建问题实例
    problem = MultiObjectiveProblem()

    # 配置算法
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.7, eta=15),
        mutation=PM(prob=0.05, eta=20),
        eliminate_duplicates=True
    )

    convergence_threshold = 0.01
    # 或者使用更智能的默认多目标终止条件
    smart_termination = MultiObjectiveDefaultTermination(
        x_tol=convergence_threshold,  # 决策变量变化容忍度
        cv_tol=convergence_threshold,  # 约束违反度变化容忍度
        f_tol=convergence_threshold,  # 目标函数变化容忍度
        nth_gen=3,  # 每5代检查一次收敛
        n_last=10,  # 考虑最近20代的进步
        n_max_gen=generations,  # 最大代数
        n_max_evals=1000  # 最大评估次数
    )

    # 执行优化
    res = minimize(problem,
                   algorithm,
                   smart_termination,
                   verbose=True,
                   seed=1)

    return res


def get_pareto_front_and_best(res):
    """获取帕累托前沿和最佳解"""
    # 获取帕累托前沿解
    F = res.F  # 目标函数值
    X = res.X  # 决策变量

    # 由于我们使用了负号来转换为最大化，这里需要转换回来
    Object_1 = -F[:, 0]  # 第一个目标：正数时间点数量

    #function4
    # Object_3 = -F[:, 2]  # 第3个目标：平均正数幅度

    # 找到最佳折衷解（基于两个目标的乘积）
    scores = Object_1
    best_index = np.argmax(scores)
    best_individual = X[best_index]


    return X, F, best_individual, best_index, Object_1


def save_results_to_excel(res, best_individual, filename="pymoo_optimization_results.xlsx"):
    global stl_filename_1
    global stl_filename_2
    global scenario_type
    """将优化结果保存到Excel文件"""

    X, F, _, best_index, Object_1= get_pareto_front_and_best(res)

    # 创建数据框存储帕累托前沿解
    pareto_data = []
    for i, (ind, ob_1) in enumerate(zip(X, Object_1)):
        result_path = "./monitor/output_"+config.ScenarioGeneratorConfig.file_vinit+".csv"
        try:
            result1 = pd.read_csv(result_path )
            time_step = len(result1)

            #


            pareto_data.append({
                'solution_index': i + 1,
                'init_velocity_1': ind[0],
                'start_1': ind[1],
                'acc_1': ind[2],
                'init_velocity_2': ind[3],
                'start_2': ind[4],
                'acc_2': ind[5],
                'Object_1': ob_1,
                #TODO replace
                'Positive_Point_Ratio(%)': np.sum((np.array(result1) > 1)) / time_step
            })
        except Exception as e:
            print(f"Error reading results for individual {i}: {e}")
            continue

    # 创建Excel写入器
    with pd.ExcelWriter(filename) as writer:
        # 保存帕累托前沿解
        if pareto_data:
            pareto_df = pd.DataFrame(pareto_data)
            pareto_df.to_excel(writer, sheet_name='pareto_front_solution', index=False)

        # 获取最佳解的时间序列
        result_path = "./monitor/output_"+config.ScenarioGeneratorConfig.file_vinit+".csv"
        try:
            result1 = pd.read_csv(result_path )

            # 保存最佳解的时间序列数据
            time_series_data = {
                'time_step': list(range(len(result1))),
                'result1': result1.values.tolist(),
                'Both_Positive': ((result1 > 1) ).astype(int).values.tolist()
            }
            time_series_df = pd.DataFrame(time_series_data)
            time_series_df.to_excel(writer, sheet_name='best_solution_timestep', index=False)

            # 保存算法参数和统计信息
            both_positive = (np.array(result1) > 1)
            positive_ratio = np.sum(both_positive) / len(result1) * 100

            stats_data = {
                'Parameter': ['init_velocity_1', 'start_1', 'init_velocity_2', 'start_2','acc_1','acc_2',
                              'Object_1', 'Positive_Point_Ratio(%)'],
                'Value': [best_individual[0], best_individual[1], best_individual[2],
                          best_individual[3], best_individual[4], best_individual[5],
                          Object_1[best_index], positive_ratio]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Best_Solution', index=False)

        except Exception as e:
            print(f"Error processing best individual results: {e}")
            traceback.print_exc()


    print(f"结果已保存到Excel文件: {filename}")
    return filename


def analyze_results(res, save_excel=True, excel_filename="pymoo_optimization_results.xlsx"):
    """分析结果并可视化"""

    # 获取帕累托前沿和最佳解
    X, F, best_individual, best_index, positive_counts = get_pareto_front_and_best(res)

    print(f"\n最佳个体参数: {best_individual}")
    print(f"目标值 - 正数时间点数: {positive_counts[best_index]}")

    # 保存结果到Excel文件
    if save_excel:
        saved_file = save_results_to_excel(res, best_individual, excel_filename)
        return X, best_individual, saved_file

    return X, best_individual


# 运行算法
if __name__ == "__main__":
    global filename
    global scenario_type


    parser = argparse.ArgumentParser("run_multi_motion_planner_system")
    parser.add_argument("--run_result", type=bool, nargs='?', const=False, default=False,
                        help="Run the generated most critical scenario with the reference planner implementation")
    parser.add_argument("--simulation_retries", type=int, nargs='?', const=10, default=10,
                        help="Amount of simulation cycles with applied mutations in between each cycle")
    parser.add_argument("--criticality_mode", type=str,
                        choices=["max_danger", "avg_danger", "random", "max_interactions"],
                        default="max_danger", help="Select a rating mode for simulations")
    parser.add_argument("--scenario_type", type=str,
                        choices=["change_lane", "crossroad", "single_straight", "roundabout", "T-intersection"],
                        default="change_lane", help="Select a scenarios type for simulations")
    parser.add_argument("--run_time", type=int, nargs='*', default=0,
                        help="run time.")
    parser.add_argument("--interaction_type", type=str,
                        choices=["merge_2", "merge_1", "cross_2", "cross_1", "cross_0", "default"],
                        default="default", help="interaction of different scenarios")
    parser.add_argument("--run_type", type=str,
                        choices=["random", "ga"],
                        default="random", help="interaction of different scenarios")
    parser.add_argument("--stl_type", type=str,
                        choices=["single", "multi"],
                        default="multi", help="single view or multi view")
    args = parser.parse_args()

    config.ScenarioGeneratorConfig.interaction_type = args.interaction_type
    config.ScenarioGeneratorConfig.run_type = args.run_type
    config.ScenarioGeneratorConfig.stl_type = args.stl_type

    scenario_type = args.scenario_type

    if args.scenario_type == "change_lane":
        filename = "USA_US101new-23_1_T-1.xml"
    elif args.scenario_type == "crossroad":
        filename = "DEU_VilaVelha-92_1_T-10.xml"
    elif args.scenario_type == "single_straight":
        filename = "USA_new3-66_1_T-10.xml"
    elif args.scenario_type == "roundabout":
        filename = "DEU_Heilbronnnew-267_1_T-5.xml"
    elif args.scenario_type == "T-intersection":
        filename = "DEU_Salzwedel-107_1_T-7.xml"

    print("开始使用pymoo进行多目标优化...")

    # 运行pymoo优化
    result = run_pymoo_optimization(
        pop_size=30,
        generations=10
    )

    # 分析结果
    pareto_front, best_solution, saved_file = analyze_results(result)


