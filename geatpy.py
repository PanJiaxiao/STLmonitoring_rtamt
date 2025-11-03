import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import subprocess
from deap import base, creator, tools, algorithms
import random
import argparse
import traceback

import config
from run_multi_motion_planner_system import get_init_scenario,ScenarioGenerator,simulate,get_random_scenario,get_mutated_scenario
import utils
from plotter.plotter import save_images_as_gif, scenario_to_image
from commonroad.scenario.lanelet import LaneletNetwork
from utils import SimulationResult, PlanningStatus, create_cc_scenario, get_cmap, prepare_scenario, write_scenario, \
    save_for_stl,clear_result
from plot import plot


def run_commonroad(vinit, start, end):
    global filename
    global scenario_type
    # command line arguments

    #print(filename)
    # #clear last result
    # clear_result("./rob_result/"+args.scenario_type)
    #TODO aviod unfulfilled

    result_init=get_init_scenario(args.scenario_type,vinit, start, end,filename)
    if result_init is False:
        # 如果function_a返回False，直接给予惩罚并跳过本次评估
        return False  # 严重惩罚
    best_scenario, best_problem_set = result_init
    # mark1
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
    # save gif of initial simulation
    sim_filename = filename.split('.')[0] + "_sim_0"
    save_images_as_gif(best_result.images, result_name, sim_filename)
    result_iteration = 0
    # initial simulation is stored at index 0
    results = [best_result]
    for simulation_number in range(1, args.simulation_retries + 1):
        try:
            # TODO often a scenario is not mutated => same criticality over and over again
            #  might be due to validation
            current_scenario, current_problem_set = best_scenario, best_problem_set
            if args.criticality_mode == "random":
                current_scenario, current_problem_set = get_random_scenario()
            else:
                current_scenario, current_problem_set = get_mutated_scenario(best_scenario, best_problem_set)
        except Exception as e:
            print(e)
            print(f"Error during scenario generation, skipping this simulation {simulation_number}!")
            continue
        try:
            current_result = simulate(current_scenario, current_problem_set, max_steps=max_steps)
        except Exception as e:
            print(e)
            print(f"Exception during simulation, skipping this simulation {simulation_number}!")
            continue
        current_avg_danger, current_danger, current_interactions, current_avg_pp_id, current_danger_pp_id = \
            utils.compute_criticality_metrics(current_result)
        current_criticality_pp_id, current_criticality = \
            utils.get_criticality_metric(avg_danger=current_avg_danger,
                                         max_danger=current_danger,
                                         max_interactions=current_interactions,
                                         max_avg_pp_id=current_avg_pp_id,
                                         max_danger_pp_id=current_danger_pp_id,
                                         mode=args.criticality_mode)
        current_result.overall_criticality = current_criticality
        current_result.simulation_number = simulation_number
        print(
            f"Simulation cycle: {simulation_number}, interactions: {current_result.interaction_counter} "
            f"criticality: {current_criticality}")

        # save gif of simulation
        sim_filename = filename.split('.')[0] + "_sim_" + str(simulation_number)
        save_images_as_gif(current_result.images, result_name, sim_filename)
        if current_criticality > best_criticality:
            best_criticality = current_criticality
            best_scenario, best_problem_set = current_scenario, current_problem_set
            best_result = current_result
            result_iteration = simulation_number

        # store results (without images to save up RAM during execution) for saving them as csv
        current_result.images = []
        results.append(current_result)

    utils.save_as_csv(results, filename, result_name)
    plot(args.scenario_type)
    # utils.save_for_stl(results, filename, result_name)
    lanelet_network: LaneletNetwork = best_scenario.lanelet_network



    # get most critical planning problem that remains in the final test scenario
    critical_key = best_criticality_pp_id
    print(f"Most critical key - criticality: {critical_key} - {best_criticality}, sim_nr: {result_iteration}")

    xml_path=write_scenario(filename, best_result.simulation_state_map, best_result.scenario, best_result.problem_set,
                   critical_key, result_name)

    save_for_stl(filename, args.scenario_type, args.run_time,xml_path,lanelet_network)

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

# 假设的函数a - 这里需要您替换为实际的函数
def function_a(parameters):
    """
    模拟函数a，接收参数，返回两个随时间变化的结果
    这里用简化的三角函数模拟，您需要替换为实际函数
    """
    # 参数示例：[振幅1, 频率1, 相位1, 振幅2, 频率2, 相位2]
    vinit1, start1, end1, vinit2, start2, end2 = parameters

    config.ScenarioGeneratorConfig.file_vinit=str(vinit1)

    print("para@@@@@@", parameters)

    result1 = []
    result2 = []

    vinit=[vinit1,vinit2]
    start=[start1,start2]
    end=[int(end1),int(end2)]

    if run_commonroad(vinit, start, end)==False:
        return False


    result_path="./rob_result/crossroad/"+config.ScenarioGeneratorConfig.file_vinit
    result1 = pd.read_csv(result_path+"/rob_save21-run_time[1].csv")
    result2 = pd.read_csv(result_path+"/rob_save22-run_time[1].csv")

    return np.array(result1), np.array(result2)


def evaluate(individual):
    """
    评估函数：计算两个目标
    目标1：两个结果都大于0的时间点数量（最大化）
    目标2：两个结果大于0的程度的平均值（最大化）
    """
    try:
        result = function_a(individual)
        if result is False:
            # 如果function_a返回False，直接给予惩罚并跳过本次评估
            return 0, -1000  # 严重惩罚
    except (IndexError, TypeError) as e:
        # 捕获IndexError，直接给予惩罚并跳过本次评估
        print(f"索引错误: {e}")
        print("IndexError or TypeError")
        traceback.print_exc()
        return 0, -1000  # 严重惩罚

    # 运行函数a
    result1, result2 = result

    # 计算两个结果都大于0的时间点
    both_positive = (result1 > 0) & (result2 > 0)
    # print("both_positive****************************",both_positive)
    positive_count = np.sum(both_positive)

    # 如果没有同时大于0的时刻，给予惩罚
    if positive_count == 0:
        return 0, -1000  # 严重惩罚

    # 计算大于0的程度（只在两个都大于0时计算）
    if positive_count > 0:
        # 取两个结果中较小的那个值，确保两个都满足大于0
        min_positive_values = np.minimum(result1[both_positive], result2[both_positive])
        positive_magnitude = np.mean(min_positive_values)
    # else:
    #     positive_magnitude = 0

    # 返回两个目标值（都是最大化）
    return positive_count, positive_magnitude


def create_toolbox():
    """创建遗传算法工具箱"""
    
    # 定义多目标优化（最大化两个目标）
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # 两个目标都是最大化
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    
    # 定义参数范围和类型
    # 6个参数：vinit1, start1, end1, vinit2, start2, end2
    #TODO change scale
    param_bounds = [
        (2.0, 8.0),    # 振幅1
        (0.0, 1.0),    # 频率1
        (0, 3),  # 相位1
        (2.0, 8.0),    # 振幅2
        (0.0, 1.0),    # 频率2
        (0, 3)   # 相位2
    ]
    
    # 注册属性生成函数
    toolbox.register("attr_float", random.uniform, 0, 1)
    
    # 注册个体和种群生成函数
    def init_individual():
        individual = []
        for bounds in param_bounds:
            individual.append(random.uniform(bounds[0], bounds[1]))
        return individual
    
    toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 注册遗传算子
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    return toolbox


def run_genetic_algorithm(pop_size=2, generations=2, cxpb=0.7, mutpb=0.3):
    """运行多目标遗传算法"""

    # clear result of each generation
    #clear_result("./rob_result/" + args.scenario_type)

    toolbox = create_toolbox()

    # 创建初始种群
    population = toolbox.population(n=pop_size)

    # 评估初始种群
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    for gen in range(generations):
    # 进化过程
        # 选择下一代
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        #TODO maybe bec the mytant
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评估新个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 更新种群
        population = offspring

        # 输出进度
        if gen % 10 == 0:
            fits = [ind.fitness.values for ind in population]
            positive_counts = [fit[0] for fit in fits]
            magnitudes = [fit[1] for fit in fits]
            print(f"Generation {gen}: Avg Positive Count = {np.mean(positive_counts):.2f}, "
                  f"Avg Magnitude = {np.mean(magnitudes):.3f}")

    return population


def get_pareto_front_and_best(population):
    """获取帕累托前沿和最佳解"""
    # 获取帕累托前沿
    # print("population@@@@@",population)
    front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    print("front!!!!!!", front)
    # 提取目标值
    positive_counts = [ind.fitness.values[0] for ind in front]
    magnitudes = [ind.fitness.values[1] for ind in front]

    # 找到最佳折衷解（基于两个目标的乘积）
    best_index = np.argmax([count * mag for count, mag in zip(positive_counts, magnitudes)])
    best_individual = front[best_index]

    return front, best_individual, best_index, positive_counts, magnitudes


def save_results_to_excel(front, best_individual, filename="genetic_algorithm_results.xlsx"):
    """将遗传算法结果保存到Excel文件"""

    # 创建数据框存储帕累托前沿解
    pareto_data = []
    for i, ind in enumerate(front):
        # result1, result2 = function_a(ind)
        print("front%%%%%%%",best_individual)
        result_path = "./rob_result/crossroad/" + str(ind[0])
        result1 = pd.read_csv(result_path + "/rob_save21-run_time[1].csv")
        result2 = pd.read_csv(result_path + "/rob_save22-run_time[1].csv")

        both_positive = (result1 > 0) & (result2 > 0)
        positive_ratio = np.sum(both_positive) / len(result1) * 100

        pareto_data.append({
            'solution_index': i + 1,
            'init_velocity_1': ind[0],
            'start_1': ind[1],
            'goal_1': ind[2],
            'init_velocity_2': ind[3],
            'start_2': ind[4],
            'goal_2': ind[5],
            'Positive_Point': ind.fitness.values[0],
            'Average_Positive_Magnitude': ind.fitness.values[1],
            'Positive_Point_Ratio(%)': positive_ratio
        })

    # 创建Excel写入器
    with pd.ExcelWriter(filename) as writer:
        # 保存帕累托前沿解
        pareto_df = pd.DataFrame(pareto_data)
        pareto_df.to_excel(writer, sheet_name='pareto_front_solution', index=False)

        # 获取最佳解的时间序列
        # result1, result2 = function_a(best_individual)
        result_path = "./rob_result/crossroad/" + str(best_individual[0])
        result1 = pd.read_csv(result_path + "/rob_save21-run_time[1].csv")
        result2 = pd.read_csv(result_path + "/rob_save22-run_time[1].csv")

        # 保存最佳解的时间序列数据
        time_series_data = {
            'time_step': list(range(len(result1))),
            'result1': result1.values.tolist(),
            'result2': result2.values.tolist(),
            'Both_Positive': ((result1 > 0) & (result2 > 0)).astype(int).values.tolist()
        }
        time_series_df = pd.DataFrame(time_series_data)
        time_series_df.to_excel(writer, sheet_name='best_solution_timestep', index=False)

        # 保存算法参数和统计信息
        both_positive = (result1 > 0) & (result2 > 0)
        positive_ratio = np.sum(both_positive) / len(result1) * 100

        stats_data = {
            'Parameter': ['init_velocity_1','start_1','goal_1','init_velocity_2','start_2','goal_2',
                     'Positive_Point', 'Average_Positive_Magnitude', 'Positive_Point_Ratio(%)'],
            'Value': [best_individual[0], best_individual[1], best_individual[2],
                   best_individual[3], best_individual[4], best_individual[5],
                   best_individual.fitness.values[0], best_individual.fitness.values[1],
                   positive_ratio]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Best_Solution', index=False)

        # 保存最佳个体参数的详细信息
        best_params_data = []

        # 基本参数信息
        best_params_data.append({
            'Parameter_Type': 'Basic_Parameter',
            'Parameter_Name': 'init_velocity_1',
            'Value': best_individual[0],
        })
        best_params_data.append({
            'Parameter_Type': 'Basic_Parameter',
            'Parameter_Name': 'start_1',
            'Value': best_individual[1],
        })
        best_params_data.append({
            'Parameter_Type': 'Basic_Parameter',
            'Parameter_Name': 'goal_1',
            'Value': best_individual[2],
        })
        best_params_data.append({
            'Parameter_Type': 'Basic_Parameter',
            'Parameter_Name': 'init_velocity_2',
            'Value': best_individual[3],

        })
        best_params_data.append({
            'Parameter_Type': 'Basic_Parameter',
            'Parameter_Name': 'start_2',
            'Value': best_individual[4],

        })
        best_params_data.append({
            'Parameter_Type': 'Basic_Parameter',
            'Parameter_Name': 'goal_2',
            'Value': best_individual[5],

        })

        # 性能指标
        best_params_data.append({
            'Parameter_Type': 'Performance_indicators',
            'Parameter_Name': 'Positive_Point',
            'Value': best_individual.fitness.values[0],
        })
        best_params_data.append({
            'Parameter_Type': 'Performance_indicators',
            'Parameter_Name': 'Average_Positive_Magnitude',
            'Value': best_individual.fitness.values[1]
        })
        best_params_data.append({
            'Parameter_Type': 'Performance_indicators',
            'Parameter_Name': 'Positive_Point_Ratio(%)',
            'Value': positive_ratio
        })

        # 统计信息
        best_params_data.append({
            'Parameter Type': 'Performance Metrics',
            'Parameter Name': 'Positive Time Points Count',
            'Parameter Value': best_individual.fitness.values[0],
            'Description': 'Number of time points where both results are greater than 0'
        })
        best_params_data.append({
            'Parameter Type': 'Performance Metrics',
            'Parameter Name': 'Average Positive Magnitude',
            'Parameter Value': best_individual.fitness.values[1],
            'Description': 'Average degree of positivity for both results'
        })
        best_params_data.append({
            'Parameter Type': 'Performance Metrics',
            'Parameter Name': 'Positive Time Points Ratio',
            'Parameter Value': positive_ratio,
            'Description': 'Percentage of time points where both results are positive (%)'
        })

        # Statistical Information
        best_params_data.append({
            'Parameter Type': 'Statistical Information',
            'Parameter Name': 'Result 1 Mean',
            'Parameter Value': np.mean(result1),
            'Description': 'Mean value of the first signal'
        })
        best_params_data.append({
            'Parameter Type': 'Statistical Information',
            'Parameter Name': 'Result 2 Mean',
            'Parameter Value': np.mean(result2),
            'Description': 'Mean value of the second signal'
        })
        best_params_data.append({
            'Parameter Type': 'Statistical Information',
            'Parameter Name': 'Result 1 Standard Deviation',
            'Parameter Value': np.std(result1),
            'Description': 'Standard deviation of the first signal'
        })
        best_params_data.append({
            'Parameter Type': 'Statistical Information',
            'Parameter Name': 'Result 2 Standard Deviation',
            'Parameter Value': np.std(result2),
            'Description': 'Standard deviation of the second signal'
        })
        best_params_data.append({
            'Parameter Type': 'Statistical Information',
            'Parameter Name': 'Result 1 Maximum',
            'Parameter Value': np.max(result1),
            'Description': 'Maximum value of the first signal'
        })
        best_params_data.append({
            'Parameter Type': 'Statistical Information',
            'Parameter Name': 'Result 2 Maximum',
            'Parameter Value': np.max(result2),
            'Description': 'Maximum value of the second signal'
        })

        # 创建并保存最佳个体参数详细信息表
        best_params_df = pd.DataFrame(best_params_data)
        best_params_df.to_excel(writer, sheet_name='best_solution', index=False)

    print(f"结果已保存到Excel文件: {filename}")
    return filename


def analyze_results(population, save_excel=True, excel_filename="genetic_algorithm_results.xlsx"):
    """分析结果并可视化"""

    # 获取帕累托前沿和最佳解
    front, best_individual, best_index, positive_counts, magnitudes = get_pareto_front_and_best(population)

    print(f"\n最佳个体参数: {best_individual}")
    print(f"目标值 - 正数时间点数: {positive_counts[best_index]}, 平均幅度: {magnitudes[best_index]:.3f}")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体，如果找不到用DejaVu Sans
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制帕累托前沿
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(positive_counts, magnitudes, alpha=0.7)
    plt.scatter(positive_counts[best_index], magnitudes[best_index], color='red', s=100, label='最佳解')
    plt.xlabel('正数时间点数量')
    plt.ylabel('平均正数幅度')
    plt.title('帕累托前沿')
    plt.legend()
    plt.grid(True)

    # 绘制最佳解的时间序列
    # plt.subplot(1, 2, 2)
    # result1, result2 = function_a(best_individual)
    # time_steps = len(result1)
    #
    # plt.plot(range(time_steps), result1, 'b-', label='结果1', linewidth=2)
    # plt.plot(range(time_steps), result2, 'r-', label='结果2', linewidth=2)
    # plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    # plt.xlabel('时间步')
    # plt.ylabel('值')
    # plt.title('最佳解的时间序列')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()
    #
    # # 统计正数时间点
    # both_positive = (result1 > 0) & (result2 > 0)
    # positive_ratio = np.sum(both_positive) / time_steps * 100
    #
    # print(f"\n性能统计:")
    # print(f"两个结果都为正的时间点比例: {positive_ratio:.1f}%")
    # print(f"结果1平均值: {np.mean(result1):.3f}")
    # print(f"结果2平均值: {np.mean(result2):.3f}")

    # 保存结果到Excel文件
    if save_excel:
        saved_file = save_results_to_excel(front, best_individual, excel_filename)
        return front, best_individual, saved_file

    return front, best_individual

    # return best_individual


# 运行算法
if __name__ == "__main__":
    global filename
    global scenario_type
    parser = argparse.ArgumentParser("run_multi_motion_planner_system")
    # parser.add_argument("--trajectory_length", type=int, nargs='?', const=20, default=20,
    #                     help="Adjust the knowledge of other planners planned trajectory.")
    parser.add_argument("--run_result", type=bool, nargs='?', const=False, default=False,
                        help="Run the generated most critical scenario with the reference planner implementation")
    parser.add_argument("--simulation_retries", type=int, nargs='?', const=10, default=10,
                        help="Amount of simulation cycles with applied mutations in between each cycle")
    parser.add_argument("--criticality_mode", type=str,
                        choices=["max_danger", "avg_danger", "random", "max_interactions"],
                        default="max_danger", help="Select a rating mode for simulations")
    parser.add_argument("--scenario_type", type=str,
                        choices=["change_lane", "crossroad", "single_straight"],
                        default="change_lane", help="Select a scenarios type for simulations")
    parser.add_argument("--run_time", type=int, nargs='*', default=0,
                        help="run time.")
    args = parser.parse_args()

    scenario_type=args.scenario_type

    if args.scenario_type == "change_lane":
        filename = "USA_US101-23_1_T-1.xml"
    elif args.scenario_type == "crossroad":
        filename = "DEU_VilaVelha-92_1_T-10.xml"
    elif args.scenario_type == "single_straight":
        filename = "USA_new-66_1_T-10.xml"
    print("开始多目标遗传算法优化...")

   # clear_result("./rob_result/" + args.scenario_type)
    # 运行遗传算法
    final_population = run_genetic_algorithm(
        pop_size=1,
        generations=5,
        cxpb=0.7,
        mutpb=0.3
    )

    # 分析结果
    # best_solution = analyze_results(final_population)
    # front, best_solution, saved_file = analyze_results(final_population)
    # # 验证最佳解
    # print("\n验证最佳解:")
    # # result1, result2 = function_a(best_solution)
    # # print("best_solution%%%%%",best_solution)
    # result_path = "./rob_result/crossroad/" + str(best_solution[0])
    # result1 = pd.read_csv(result_path + "/rob_save21-run_time[1].csv")
    # result2 = pd.read_csv(result_path + "/rob_save22-run_time[1].csv")
    # print(f"结果1 > 0 的时间点: {np.sum(result1 > 0)}/{len(result1)}")
    # print(f"结果2 > 0 的时间点: {np.sum(result2 > 0)}/{len(result1)}")
    # print(f"结果1 > 0 的平均幅度: {np.mean(result1[result1 > 0]) if np.any(result1 > 0) else 0:.3f}")
    # print(f"结果2 > 0 的平均幅度: {np.mean(result2[result2 > 0]) if np.any(result2 > 0) else 0:.3f}")
    # print(f"两个结果都 > 0 的时间点: {np.sum((result1 > 0) & (result2 > 0))}/{len(result1)}")