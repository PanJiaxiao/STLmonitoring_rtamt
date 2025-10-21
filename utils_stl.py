import operator
from copy import deepcopy
from enum import Enum
import os
import math
import csv
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
import random

import matplotlib.pyplot as plt

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import LaneletNetwork,Lanelet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType


from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.geometry.shape import Rectangle
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile

from commonroad_dc.boundary.boundary import create_road_boundary_obstacle
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker

import commonroad.geometry.shape as shp
from tensorflow_core.python.ops.gen_bitwise_ops import left_shift

from planners.reactive_planner.commonroad_rp.parameter import VehModelParameters

from config import ScenarioGeneratorConfig
from planning_generator import PlanningGenerator
from scenario_generator.scenario_generator import ScenarioGenerator
from plotter.plotter import result_path
from validation.validator import get_successor_lanelets_notmerge ,get_predecessor_lanelets_notmerge



import scipy.io



def save_as_mat(matrix:np.ndarray, filename:str,file_index: str, phi_str: str,signal_str: str):
    # double_matrix_2 = matrix.astype(np.float64)
    output_filename = filename.split('.')[0] + '-multi-planner-simulation-result-'+file_index+'.mat'  
    scipy.io.savemat(output_filename, {'phi_str':phi_str,
                                        'signal_str':signal_str,'tau':0,'trace': matrix})
    print("数据已保存")


def compute_lan_orientaion(laneletnetwork:LaneletNetwork, position:float,orientation:float):
    lan_o_1=0
    for lanelet in laneletnetwork.lanelets:
        if lanelet.polygon.contains_point(position):
            lan_o_1=abs(orientation_by_position(lanelet,position)-orientation)

    return lan_o_1

def compute_distance(matrix_2:np.ndarray,max_length_2:int,a_x_index:int,a_y_index:int,b_x_index:int,b_y_index:int):
    distance_2=list()
    for i in range(0,max_length_2-1):
        a_x=matrix_2[a_x_index][i]
        a_y=matrix_2[a_y_index][i]
        b_x=matrix_2[b_x_index][i]
        b_y=matrix_2[b_y_index][i]
        if a_x!=0 and a_y!=0 and b_x!=0 and b_y!=0:
            dis_x=math.pow(a_x,2)-math.pow(b_x,2)
            dis_y=math.pow(a_y,2)-math.pow(b_y,2)
            distance_2.append(math.sqrt(abs(dis_x-dis_y)))
    #填充distance，转换成nparray
    padded_distance_2 = distance_2 + [0] * (max_length_2 - len(distance_2))
    # matrix_2 = np.vstack((matrix_2, padded_distance_2))

    return padded_distance_2

def compute_acc(matrix_2:np.ndarray,max_length_2:int,v_index:int):
    acc = []
    for i in range(1,max_length_2-1):
        if matrix_2[v_index][i]!=0:
            if matrix_2[v_index][i]>matrix_2[v_index][i-1]:
                acc.append(1)
            else :
                acc.append(-1)
    padded_va_acc = acc + [0] * (max_length_2 - len(acc))
    # matrix_2 = np.vstack((matrix_2, padded_va_acc))

    return padded_va_acc

def compute_a_bef_b(laneletnetwork:LaneletNetwork, position_a,position_b,scenario_type):
    #a_bef_b
    if scenario_type=="change_lane":
        lanelet = random.choice(laneletnetwork.lanelets)
        first_lanelets= get_predecessor_lanelets_notmerge(lanelet, laneletnetwork)
        succ_lanelets = get_successor_lanelets_notmerge(lanelet, laneletnetwork)
        first_lanelets.append(lanelet)
        first_lanelets.extend(succ_lanelets)
        #lanelets=pre-lan-succ
        while(lanelet in first_lanelets):
            lanelet = random.choice(laneletnetwork.lanelets)
        seconed_lanelets = get_predecessor_lanelets_notmerge(lanelet, laneletnetwork)
        succ_lanelets = get_successor_lanelets_notmerge(lanelet, laneletnetwork)
        seconed_lanelets.append(lanelet)
        seconed_lanelets.extend(succ_lanelets)

        left_boundary_1 = first_lanelets[0].left_vertices
        right_boundary_1 = first_lanelets[0].right_vertices
        for i in range(len(first_lanelets)-1):
            #index smaller=>closer to the start
            left_boundary_1 = np.concatenate((left_boundary_1, first_lanelets[i+1].left_vertices), axis=0)
            right_boundary_1 = np.concatenate((right_boundary_1, first_lanelets[i+1].right_vertices), axis=0)

        left_boundary_2 = seconed_lanelets[0].left_vertices
        right_boundary_2 = seconed_lanelets[0].right_vertices
        for i in range(len(seconed_lanelets)-1):
            left_boundary_2 = np.concatenate((left_boundary_2, seconed_lanelets[i+1].left_vertices), axis=0)
            right_boundary_2 = np.concatenate((right_boundary_2, seconed_lanelets[i+1].right_vertices), axis=0)


        polygon_1 = []
        for i in range(len(left_boundary_1)-1):
            #polygon i terminat point index=>i+1
            left=np.array([left_boundary_1[i],left_boundary_1[i+1]])
            right=np.array([right_boundary_1[i],right_boundary_1[i+1]])
            polyline=np.concatenate((left,np.flipud(right)))
            polygon=shp.Polygon(polyline)
            polygon_1.append(polygon)
        polygon_2 = []
        for i in range(len(left_boundary_2)-1):
            left=np.array([left_boundary_2[i],left_boundary_2[i+1]])
            right=np.array([right_boundary_2[i],right_boundary_2[i+1]])
            polyline=np.concatenate((left,np.flipud(right)))
            polygon=shp.Polygon(polyline)
            polygon_2.append(polygon)

        lan_position_a=[]
        index_a=[]
        for i in range(len(polygon_1)-1):
            if polygon_1[i].contains_point(position_a) or polygon_2[i].contains_point(position_a):
                lan_position_a.append(i)
                index_a.append(i)
            else:
                lan_position_a.append(-1)

        lan_position_b=[]
        for i in range(len(polygon_1)-1):
            if polygon_1[i].contains_point(position_b) or polygon_2[i].contains_point(position_b):
                lan_position_b.append(i)
            else:
                lan_position_b.append(-1)

        a_bef_b = []
        # 应该以时间step为标准，而不是道路网格
        if lan_position_a[-1] != -1 and lan_position_b[-1] != -1:
            if lan_position_a[-1] > lan_position_b[-1]:
                a_bef_b.append(1)
            elif lan_position_a[i] == lan_position_b[-1]:
                # which ego car closer to the first point of this
                left_end_1 = left_boundary_1[index_a[-1] + 1]
                right_end_1 = right_boundary_1[index_a[-1] + 1]
                left_end_2 = left_boundary_2[index_a[-1] + 1]
                right_end_2 = right_boundary_2[index_a[-1] + 1]
                center_x_1 = left_end_1[0]+right_end_1[0]
                center_y_1 = left_end_1[1]+right_end_1[1]
                center_x_2 = left_end_2[0]+right_end_2[0]
                center_y_2 = left_end_2[1] + right_end_2[1]

                center_1=[center_x_1,center_y_1]
                center_2=[center_x_2,center_y_2]

                dis_a = min(math.dist(position_a, center_1), math.dist(position_a, center_2))
                dis_b = min(math.dist(position_b, center_1), math.dist(position_b, center_2))

                if dis_a>dis_b:
                    a_bef_b.append(-1)
                elif dis_a<dis_b:
                    a_bef_b.append(1)
                else:
                    a_bef_b.append(0)

            else:
                a_bef_b.append(-1)
        else:
            a_bef_b.append(100)

    elif scenario_type=="single_straight":
        lanelet = random.choice(laneletnetwork.lanelets)
        first_lanelets = get_predecessor_lanelets_notmerge(lanelet, laneletnetwork)
        succ_lanelets = get_successor_lanelets_notmerge(lanelet, laneletnetwork)
        first_lanelets.append(lanelet)
        first_lanelets.extend(succ_lanelets)

        left_boundary_1 = first_lanelets[0].left_vertices
        right_boundary_1 = first_lanelets[0].right_vertices
        for i in range(len(first_lanelets) - 1):
            left_boundary_1 = np.concatenate((left_boundary_1, first_lanelets[i + 1].left_vertices), axis=0)
            right_boundary_1 = np.concatenate((right_boundary_1, first_lanelets[i + 1].right_vertices), axis=0)

        polygon_1 = []
        for i in range(len(left_boundary_1) - 1):
            left = np.array([left_boundary_1[i], left_boundary_1[i + 1]])
            right = np.array([right_boundary_1[i], right_boundary_1[i + 1]])
            polyline = np.concatenate((left, np.flipud(right)))
            polygon = shp.Polygon(polyline)
            polygon_1.append(polygon)

        lan_position_a = []
        index_a=[]
        for i in range(len(polygon_1) - 1):
            if polygon_1[i].contains_point(position_a) :
                lan_position_a.append(i)
                index_a.append(i)
            else:
                lan_position_a.append(-1)

        lan_position_b = []
        for i in range(len(polygon_1) - 1):
            if polygon_1[i].contains_point(position_b) :
                lan_position_b.append(i)
            else:
                lan_position_b.append(-1)

        a_bef_b = []
        # 应该以时间step为标准，而不是道路网格
        if lan_position_a[-1] != -1 and lan_position_b[-1] != -1:
            if lan_position_a[-1] > lan_position_b[-1]:
                a_bef_b.append(1)
            elif lan_position_a[i] == lan_position_b[-1]:
                # which ego car closer to the first point of this
                left_end_1 = left_boundary_1[index_a[-1] + 1]
                right_end_1 = right_boundary_1[index_a[-1] + 1]

                center_x_1 = left_end_1[0] + right_end_1[0]
                center_y_1 = left_end_1[1] + right_end_1[1]

                center_1 = [center_x_1, center_y_1]

                dis_a = math.dist(position_a, center_1)
                dis_b = math.dist(position_b, center_1)

                if dis_a > dis_b:
                    a_bef_b.append(-1)
                elif dis_a < dis_b:
                    a_bef_b.append(1)
                else:
                    a_bef_b.append(0)
                a_bef_b.append(0)
            else:
                a_bef_b.append(-1)
        else:
            a_bef_b.append(100)

    return a_bef_b[-1]


def orientation_by_position(lanet:Lanelet, position: np.ndarray) -> float:
        """
        Returns lanelet orientation closest to a given position
        :param position: position of interest
        :return: orientation in interval [-pi,pi]
        """

        def check_angle(point, point1, point2):
            vector_1 = point - point1
            vector_2 = point2 - point1

            def unit_vector(vector):
                norm = np.linalg.norm(vector)
                if np.isclose(norm, 0.0):
                    return vector
                else:
                    return vector / norm

            dot_product = np.dot(unit_vector(vector_1), unit_vector(vector_2))
            return np.rad2deg(np.arccos(dot_product))

        assert (
            check_angle(position, lanet.center_vertices[-1], lanet.center_vertices[-2]) <= 90
            and check_angle(position, lanet.center_vertices[0], lanet.center_vertices[1]) <= 90
        )
        position_diff_square = np.sum((lanet.center_vertices - position) ** 2, axis=1)

        closest_vertex_index = np.argmin(position_diff_square)

        if closest_vertex_index == len(lanet.center_vertices) - 1:
            vertex1 = lanet.center_vertices[closest_vertex_index - 1, :]
            vertex2 = lanet.center_vertices[closest_vertex_index, :]
        else:
            vertex1 = lanet.center_vertices[closest_vertex_index, :]
            vertex2 = lanet.center_vertices[closest_vertex_index + 1, :]

        direction_vector = vertex2 - vertex1

        return np.arctan2(direction_vector[1], direction_vector[0])
