#!/bin/bash

# 自动运行 run_multi_motion_planner_system.py 10次的脚本

# 设置conda环境名称
CONDA_ENV="croad"

# 初始化conda
eval "$(conda shell.bash hook)"

# 激活conda环境
conda activate "$CONDA_ENV"

if [ $? -ne 0 ]; then
    echo "错误: 无法激活conda环境 '$CONDA_ENV'"
    exit 1
fi

echo "已激活conda环境: $CONDA_ENV"
echo "Python路径: $(which python)"

# 设置Python解释器
PYTHON="python3"

# 设置要运行的Python文件路径
SCRIPT_FILE="run_multi_motion_planner_system.py"

# 设置运行次数
RUN_TIMES=80

# 检查文件是否存在
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "错误: 找不到文件 $SCRIPT_FILE"
    echo "请确保脚本在当前目录或提供完整路径"
    exit 1
fi

echo "开始运行 $SCRIPT_FILE 共 $RUN_TIMES 次"
echo "参数: --simulation_retries 0 --criticality_mode max_danger --scenario_type single_straight"
echo "================================================"

# 记录开始时间
START_TIME=$(date +%s)

# 循环运行10次
for ((i=1; i<=$RUN_TIMES; i++))
do
    echo "第 $i 次运行开始..."
    echo "----------------------------------------"
    
    # 运行Python脚本并传递参数
    $PYTHON "$SCRIPT_FILE" \
        --simulation_retries 0 \
        --criticality_mode max_danger \
        --scenario_type crossroad \
        --run_time $i
    
    # 检查运行状态
    if [ $? -eq 0 ]; then
        echo "第 $i 次运行成功完成!"
    else
        echo "第 $i 次运行失败，退出代码"
        #echo "是否继续运行？(y/n)"
        #read -r response
        #if [[ ! "$response" =~ ^[Yy]$ ]]; then
         #   echo "用户选择终止运行"
          #  exit 1
        #fi
    fi
    
    echo "----------------------------------------"
    echo ""
    
    # 可选：添加延迟，避免连续运行过于密集
    # sleep 1
done

# 记录结束时间并计算总耗时
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "================================================"
echo "所有运行完成!"
echo "总运行次数: $RUN_TIMES"
echo "总耗时: $TOTAL_TIME 秒"
echo "平均每次运行时间: $((TOTAL_TIME / RUN_TIMES)) 秒"
