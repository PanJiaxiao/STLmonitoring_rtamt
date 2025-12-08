import os
from commonroad.common.file_reader import CommonRoadFileReader

if __name__ == "__main__":
    # 1. 替换为你的CommonRoad场景文件路径
    filename ="./results/DEU_VilaVelha-92_1_T-10_max_danger_crossroad_[1]_5.907735016423407/DEU_VilaVelha-92_1_T-10-multi-planner-simulate.xml"

    # 2. 读取场景
    scenario, planning_problem_set = CommonRoadFileReader(filename).open()

    # 3. 获取并打印时间步长dt
    dt = scenario.dt  # 单位：秒
    print(f"该场景的时间步长 dt = {dt} 秒")
    print(f"即每个 time_step 对应 {dt * 1000} 毫秒")