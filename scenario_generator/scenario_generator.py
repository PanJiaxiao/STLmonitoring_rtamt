from copy import deepcopy
from typing import Tuple,List
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import operator


import networkx as nx

from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.trajectory import State
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from commonroad.scenario.scenario import Scenario
from commonroad.common.util import Interval
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

from commonroad.visualization.mp_renderer import MPRenderer

from shapely.geometry import LineString, Point, Polygon
from commonroad_route_planner.route_planner import RoutePlanner
from tensorflow_core.python import empty

import config
from validation.validator import cut, get_successor_lanelets ,get_straight_lanelets,get_predecessor_lanelets_notmerge,get_successor_lanelets_notmerge
from config import ScenarioGeneratorConfig
import utils

#filled by gen_income,income is index of lan's center_vertices
crossroad_income=[]
#end is index of lan;goal is terminal of a lan
crossroad_outcome=[36,1587,40,1583]
income_list=[1584,37,1586,39]
#stl=[21,22]
goal_id=40
# goal_id=[40,1587]
start_id=[]


class ScenarioGenerator:
    def __init__(self, scenario: Scenario):
        self.config: ScenarioGeneratorConfig = ScenarioGeneratorConfig()
        self.scenario = scenario
        self.lanelet_network = scenario.lanelet_network


    #generate income
    def gen_income(self,scenario: Scenario,lan_id):
        # global income_list
        global crossroad_income
        crossroad_income.clear()
        # for i in income_list:
            #print("incom_list",i)
        # for i in lan_id:
        lan=scenario.lanelet_network.find_lanelet_by_id(lan_id)
        lan_center=lan.center_vertices.tolist()
        #print("lan_center",lan_center)
        crossroad_income=crossroad_income+lan_center

    def gen_start_and_goal_id(self,scenario_type,interaction_type):
        global goal_id
        global start_id

        if scenario_type=="crossroad":
            if interaction_type=="merge_2":
                # print("#####arrive startgen")
                goal_id=[36,36]
                start_id=[1584,1586]
            elif interaction_type=="merge_1":
                start_id=[39,1586]
                goal_id=[36,36]
            elif interaction_type=="cross_2":
                start_id=[37,1584]
                goal_id=[1583,40]
            elif interaction_type=="cross_1":
                start_id=[37,1584]
                goal_id = [1583,1587]
            elif interaction_type=="cross_0":
                start_id = [1584, 37]
                goal_id = [1587, 40]
        elif scenario_type=="single_straight":
            start_id = [91, 91]
            goal_id = [91, 91]
        elif scenario_type == "change_lane":
            start_id = [137148, 137143]
            goal_id = [137143, 137148]
        elif scenario_type == "roundabout":
            if interaction_type=="merge_1":
                start_id=[12451,3802]
                goal_id=[3471,3471]
            elif interaction_type=="cross_0":
                start_id = [4822, 3802]
                goal_id = [11467, 3471]
        if scenario_type == "T-intersection":
            if interaction_type == "merge_2":
                goal_id = [616, 616]
                start_id = [581, 583]
            elif interaction_type == "merge_1":
                start_id = [617,581]
                goal_id = [584,584]
            elif interaction_type == "cross_2":
                start_id = [581, 617]
                goal_id = [616, 584]
            elif interaction_type == "cross_1":
                start_id = [617,583]
                goal_id = [584,580]


    #mark5
    def generate(self, scenario: Scenario, problem_set: PlanningProblemSet,scenario_type):
        global goal_id
        global start_id
        self.gen_start_and_goal_id(scenario_type,config.ScenarioGeneratorConfig.interaction_type)
        vinit=config.ScenarioGeneratorConfig.vinit
        start=config.ScenarioGeneratorConfig.start
        acc=config.ScenarioGeneratorConfig.acc
        if scenario_type=="change_lane":
            if config.ScenarioGeneratorConfig.run_type == "random":
                #config中的pp数量
                print("arrive_random_init")
                first_lanlets=scenario.lanelet_network.find_lanelet_by_id(137148)
                seconde_lanlets=scenario.lanelet_network.find_lanelet_by_id(137143)
                for i in range(self.config.amount_pp):
                    if i==0:
                        goal_area = self.get_possible_goal_for_lanelet(seconde_lanlets)
                        init_state = self.get_random_init_state(first_lanlets)
                    else :
                        goal_area = self.get_possible_goal_for_lanelet(first_lanlets)
                        init_state = self.get_random_init_state(seconde_lanlets)
                    goal = self.generate_goal(goal_area)
                    id = 999 + i
                    new_pp = PlanningProblem(id, init_state, goal)
                    problem_set.add_planning_problem(new_pp)
            else:
                for i in range(self.config.amount_pp):

                    goal_area = self.get_possible_goal_for_lanelet(scenario.lanelet_network.find_lanelet_by_id(goal_id[i]))

                    goal = self.generate_goal(goal_area)

                    id = 999 + i
                    #TODO
                    init_state = self.get_ga_init_state(scenario,vinit[i], start[i],start_id[i],acc[i])
                    new_pp = PlanningProblem(id, init_state, goal)
                    problem_set.add_planning_problem(new_pp)
        elif scenario_type=="crossroad":
            if config.ScenarioGeneratorConfig.run_type=="random":
                for i in range(self.config.amount_pp):

                    #stl{21,22}-one goal
                    goal_area = self.get_possible_goal_for_lanelet(scenario.lanelet_network.find_lanelet_by_id(goal_id[i]))

                    goal = self.generate_goal(goal_area)

                    id = 999 + i
                    #TODO
                    lanelet=scenario.lanelet_network.find_lanelet_by_id(start_id[i])
                    init_state = self.get_random_init_state(lanelet)
                    new_pp = PlanningProblem(id, init_state, goal)
                    problem_set.add_planning_problem(new_pp)
                # incoming_map = scenario.lanelet_network.intersections[0].map_incoming_lanelets
                # first_id = []
                # for i in range(self.config.amount_pp):
                #     incoming_id = random.sample(list(incoming_map), 1)
                #     while incoming_id in first_id:
                #         incoming_id = random.sample(list(incoming_map), 1)
                #     first_id.append(incoming_id)
                #     incoming_ele = incoming_map[incoming_id[0]]
                #     lanelet = scenario.lanelet_network.find_lanelet_by_id(incoming_id[0])
                #     # TODO diff start area
                #     goal_lanelets = incoming_ele.successors_right | incoming_ele.successors_left | incoming_ele.successors_straight
                #     #set().pop radom throw out
                #     goal_lanelet_id = goal_lanelets.pop()
                #     goal_lanelet = scenario.lanelet_network.find_lanelet_by_id(goal_lanelet_id)
                #     succ_lanelets = get_successor_lanelets_notmerge(
                #         goal_lanelet, scenario.lanelet_network)
                #     goal_area = self.get_possible_goal_for_lanelet(random.choice(succ_lanelets))
                #     goal = self.generate_goal(goal_area)
                #     id = 999 + i
                #
                #     init_state = self.get_random_init_state(lanelet)
                #     new_pp = PlanningProblem(id, init_state, goal)
                #     problem_set.add_planning_problem(new_pp)
            else:
                for i in range(self.config.amount_pp):

                    #stl{21,22}-one goal
                    goal_area = self.get_possible_goal_for_lanelet(scenario.lanelet_network.find_lanelet_by_id(goal_id[i]))

                    goal = self.generate_goal(goal_area)

                    id = 999 + i
                    #TODO
                    init_state = self.get_ga_init_state(scenario,vinit[i], start[i],start_id[i],acc[i])
                    new_pp = PlanningProblem(id, init_state, goal)
                    problem_set.add_planning_problem(new_pp)

        elif scenario_type == "single_straight":
            if config.ScenarioGeneratorConfig.run_type=="random":
                #a_bef_b
                for i in range(self.config.amount_pp):
                    lanelet =scenario.lanelet_network.find_lanelet_by_id(91)
                    # a must before b
                    init_state= self.get_random_init_state(lanelet)
                    goal_area = self.get_possible_goal_for_lanelet(lanelet)
                    goal = self.generate_goal(goal_area)
                    id = 999 + i

                    new_pp = PlanningProblem(id, init_state, goal)
                    problem_set.add_planning_problem(new_pp)
            else:
                for i in range(self.config.amount_pp):
                    # stl{21,22}-one goal
                    goal_area = self.get_possible_goal_for_lanelet(
                        scenario.lanelet_network.find_lanelet_by_id(goal_id[i]))
                    goal = self.generate_goal(goal_area)
                    id = 999 + i
                    # TODO
                    init_state = self.get_ga_init_state(scenario, vinit[i], start[i], start_id[i], acc[i])
                    new_pp = PlanningProblem(id, init_state, goal)
                    problem_set.add_planning_problem(new_pp)

        elif scenario_type == "roundabout":
            if config.ScenarioGeneratorConfig.run_type=="random":
                for i in range(self.config.amount_pp):
                    # stl{21,22}-one goal
                    goal_area = self.get_possible_goal_for_lanelet(
                        scenario.lanelet_network.find_lanelet_by_id(goal_id[i]))

                    goal = self.generate_goal(goal_area)

                    id = 999 + i
                    # TODO
                    lanelet=scenario.lanelet_network.find_lanelet_by_id(start_id[i])
                    init_state = self.get_random_init_state(lanelet)
                    new_pp = PlanningProblem(id, init_state, goal)
                    problem_set.add_planning_problem(new_pp)
            else:
                for i in range(self.config.amount_pp):

                    #stl{21,22}-one goal
                    goal_area = self.get_possible_goal_for_lanelet(scenario.lanelet_network.find_lanelet_by_id(goal_id[i]))

                    goal = self.generate_goal(goal_area)

                    id = 999 + i
                    #TODO
                    init_state = self.get_ga_init_state(scenario,vinit[i], start[i],start_id[i],acc[i])
                    new_pp = PlanningProblem(id, init_state, goal)
                    problem_set.add_planning_problem(new_pp)
        elif scenario_type == "T-intersection":
            if config.ScenarioGeneratorConfig.run_type=="random":
                for i in range(self.config.amount_pp):
                    # stl{21,22}-one goal
                    goal_area = self.get_possible_goal_for_lanelet(
                        scenario.lanelet_network.find_lanelet_by_id(goal_id[i]))

                    goal = self.generate_goal(goal_area)

                    id = 999 + i
                    # TODO
                    lanelet=scenario.lanelet_network.find_lanelet_by_id(start_id[i])
                    init_state = self.get_random_init_state(lanelet)
                    new_pp = PlanningProblem(id, init_state, goal)
                    problem_set.add_planning_problem(new_pp)
            else:
                for i in range(self.config.amount_pp):

                    #stl{21,22}-one goal
                    goal_area = self.get_possible_goal_for_lanelet(scenario.lanelet_network.find_lanelet_by_id(goal_id[i]))

                    goal = self.generate_goal(goal_area)

                    id = 999 + i
                    #TODO
                    init_state = self.get_ga_init_state(scenario,vinit[i], start[i],start_id[i],acc[i])
                    new_pp = PlanningProblem(id, init_state, goal)
                    problem_set.add_planning_problem(new_pp)
        return scenario, problem_set

    def get_dynamic_obstacle(self,scenario:Scenario,straight_lanets:List[Lanelet]):
        # 查找lanelet  
        lanelet = random.choice(straight_lanets)

        # initial state has a time step of 0
        # dynamic_obstacle_initial_state = State(position = lanelet.center_vertices[0],
        #                                             velocity = 15,
        #                                             orientation = self.get_orientation(lanelet.left_vertices[0],lanelet.right_vertices[0]),
        #                                             time_step = 0)
        dynamic_obstacle_initial_state=self.get_random_init_state(lanelet)

        # generate the states for the obstacle for time steps 1 to 40 by assuming constant velocity
        state_list = []
        new_position=lanelet.center_vertices[0]
        for i in range(1, 10):
            # compute new position
            orientation=self.get_orientation(lanelet.left_vertices[0],lanelet.right_vertices[0])
            new_position = self.calculate_next_position(new_position,15,orientation)
            # create new state
            new_state = State(position = new_position, velocity = 15,orientation = orientation, time_step = i)
            # add new state to state_list
            state_list.append(new_state)

        # create the trajectory of the obstacle, starting at time step 1
        dynamic_obstacle_trajectory = Trajectory(1, state_list)

        # create the prediction using the trajectory and the shape of the obstacle
        dynamic_obstacle_shape = Rectangle(width = 5.0, length = 10.0)
        dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)

        # generate the dynamic obstacle according to the specification
        dynamic_obstacle_id = scenario.generate_object_id()
        dynamic_obstacle_type = ObstacleType.CAR
        dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                        dynamic_obstacle_type,
                                        dynamic_obstacle_shape,
                                        dynamic_obstacle_initial_state,
                                        dynamic_obstacle_prediction)

        # add dynamic obstacle to the scenario
        scenario.add_objects(dynamic_obstacle)

        return scenario

    def get_orientation(self,right_vertex:tuple,left_vertex:tuple):
        dx = right_vertex[0] - left_vertex[0]
        dy = right_vertex[1] - left_vertex[1]

        # 计算朝向  
        orientation = np.arctan2(dy, dx)
        return orientation

    def calculate_next_position(self,current_position: tuple, speed: float, orientation: float) -> tuple:
        # 计算速度的 x 和 y 分量  
        velocity_x = speed * np.cos(orientation)
        velocity_y = speed * np.sin(orientation)

        # 更新位置  
        new_x = current_position[0] + velocity_x
        new_y = current_position[1] + velocity_y

        return (new_x, new_y)

    def get_random_scenario(self, scenario: Scenario, problem_set: PlanningProblemSet):
        # pp_to_add = random.randint(self.config.min_amount_pp, self.config.max_amount_pp)
        pp_to_add=2
        for i in range(0, pp_to_add):
            self.add_random_planning_problem(scenario, problem_set)
        self.mutate_scenario(scenario, problem_set)
        return scenario, problem_set

    def mutate_scenario(self, scenario: Scenario, problem_set: PlanningProblemSet):
        if not self.config.do_mutations:
            return
        while self.config.min_amount_pp - len(problem_set.planning_problem_dict.values()) > 0:
            self.add_random_planning_problem(scenario, problem_set)
        number_pp = len(problem_set.planning_problem_dict.values())
        is_mutation_applied = False
        # TODO could also mutate all the problems instead of one at a time (on average)
        for pp in list(problem_set.planning_problem_dict.values()):
            random_number = random.uniform(0, 1)
            if random_number < 1 / number_pp:
                other_random = random.uniform(0, 1)
                # due to 5 applied mutation operators
                amount_mutation_operators = 5
                if other_random < 1 / amount_mutation_operators:
                    is_mutation_applied = True
                    self.mutate_planning_problem_goal(pp)
                other_random = random.uniform(0, 1)
                if other_random < 1 / amount_mutation_operators:
                    is_mutation_applied = self.mutate_planning_problem_velocity(pp)
                other_random = random.uniform(0, 1)
                if other_random < 1 / amount_mutation_operators:
                    is_mutation_applied = True
                    self.mutate_planning_problem_position(pp, scenario)
                other_random = random.uniform(0, 1)
                if other_random < 1 / amount_mutation_operators and number_pp < self.config.max_amount_pp:
                    is_mutation_applied = True
                    self.add_random_planning_problem(scenario, problem_set)
                other_random = random.uniform(0, 1)
                if other_random < 1 / amount_mutation_operators and number_pp > self.config.min_amount_pp:
                    is_mutation_applied = True
                    ScenarioGenerator.remove_random_planning_problem(problem_set)
        if not is_mutation_applied:
            self.mutate_scenario(scenario, problem_set)

    def mutate_planning_problem_position(self, planning_problem: PlanningProblem, scenario: Scenario):
        current_position = planning_problem.initial_state.position
        ll_network = scenario.lanelet_network
        lanelets = ll_network.find_lanelet_by_position([current_position])
        # planning_problem is initially offroad
        if len(lanelets) == 0 or len(lanelets[0]) == 0:
            return
        lanelet: Lanelet = ll_network.find_lanelet_by_id(lanelets[0][0])
        center_line = LineString(lanelet.center_vertices)
        position_modifier = random.uniform(
            -self.config.max_position_modifier, self.config.max_position_modifier)
        new_position = center_line.interpolate(position_modifier)
        planning_problem.initial_state.position = np.array([
            new_position.x, new_position.y])

    def mutate_planning_problem_velocity(self, planning_problem: PlanningProblem):
        current_velocity = planning_problem.initial_state.velocity
        velocity_modifier = random.uniform(
            -self.config.max_init_velocity, self.config.max_init_velocity)
        new_velocity = current_velocity + velocity_modifier
        # TODO velocity sometimes is negative; might come from here
        if self.config.min_velocity <= new_velocity <= self.config.max_velocity:
            planning_problem.initial_state.velocity = new_velocity
            return True
        else:
            return False

    def mutate_planning_problem_goal(self, planning_problem: PlanningProblem):
        # might have problems with circular road networks due to the check for lanelet.successor == 0
        exit_points = set(
            [lanelet.lanelet_id for lanelet in self.lanelet_network.lanelets if len(lanelet.successor) == 0])
        route_planner = RoutePlanner(
            self.scenario, planning_problem)
        # force graph generation (for survival scnearios without goal area the route_planner does not create a graph per default)
        graph = route_planner._create_graph_from_lanelet_network()
        initial_lanelets = self.lanelet_network.find_lanelet_by_position(
            [planning_problem.initial_state.position])
        if len(initial_lanelets) == 0 or len(initial_lanelets[0]) == 0:
            # could not find lanelet for init position
            return
        initial_lanelet_id = initial_lanelets[0][0]
        # make sure to only choose from reachable lanelets
        reachable_lanelets = nx.descendants(graph, initial_lanelet_id)
        if len(reachable_lanelets) == 0:
            reachable_lanelets.add(initial_lanelet_id)
        leaf_nodes = list(exit_points.intersection(reachable_lanelets))
        # TODO improvement: try to choose a different goal area than previously
        #  or use assertion that it could not be mutated to a reachable goal area => maybe validation
        ll_id = random.choice(leaf_nodes)
        lanelet = self.lanelet_network.find_lanelet_by_id(ll_id)
        goal_area = self.get_possible_goal_for_lanelet(lanelet)
        goal = self.generate_goal(goal_area)

        planning_problem.goal = goal

    def add_random_planning_problem(self, scenario: Scenario, problem_set: PlanningProblemSet,choiced_lanelets: Lanelet):
        #随机选择车道段
        lanelet = random.choice(scenario.lanelet_network.lanelets)
        #不选择重复的车道
        if choiced_lanelets:
            while(lanelet in choiced_lanelets):
                lanelet = random.choice(scenario.lanelet_network.lanelets)
        #获取后续车道段
        full_lanelets = get_successor_lanelets(
            lanelet, scenario.lanelet_network)

        choiced_lanelets=full_lanelets

        #选择目标区域
        goal_area = self.get_possible_goal_for_lanelet(random.choice(full_lanelets))
        goal = self.generate_goal(goal_area)
        if problem_set.planning_problem_dict:
            last_key, pp = max(list(problem_set.planning_problem_dict.items()))
        else:
            last_key = 0
        new_id = last_key + 1
        init_state = self.get_random_init_state(random.choice(full_lanelets))
        new_pp = PlanningProblem(new_id, init_state, goal)
        problem_set.add_planning_problem(new_pp)

        return full_lanelets

    @staticmethod
    def remove_random_planning_problem(problem_set: PlanningProblemSet):
        pp = random.choice(list(problem_set.planning_problem_dict.values()))
        pp_id_to_remove = problem_set.find_planning_problem_by_id(pp.planning_problem_id)
        problem_set.planning_problem_dict.pop(pp_id_to_remove.planning_problem_id)

    #mark6

    def get_random_init_state(self, lanelet: Lanelet):
        # mandatory fields for init State: [position, velocity, orientation, yaw_rate, slip_angle, time_step]
        #random
        velocity = random.uniform(
            self.config.min_init_velocity, self.config.max_init_velocity)

        acceleration = random.uniform(
            self.config.min_init_acceleration, self.config.max_init_acceleration)

        random_index = random.choice(range(len(lanelet.center_vertices) - 1))
        position = lanelet.center_vertices[random_index]

        # TODO might be out of bounds
        next_point = lanelet.center_vertices[random_index + 1]
        orientation = get_orientation_by_coords(
            (position[0], position[1]), (next_point[0], next_point[1]))

        yaw_rate = 0.0
        slip_angle = 0.0

        return State(velocity=velocity, orientation=orientation, time_step=0, position=position,acceleration=acceleration, yaw_rate=yaw_rate,
                     slip_angle=slip_angle)

    def get_ga_init_state(self,scenario: Scenario,vint,index,lan_id,acc):
        global crossroad_income
        global income_list
        # mandatory fields for init State: [position, velocity, orientation, yaw_rate, slip_angle, time_step]

        velocity=vint
        acceleration=acc

        self.gen_income(scenario,lan_id)

        #fix the start
        lanelet = scenario.lanelet_network.find_lanelet_by_id(lan_id)
        #index[0,1];crossroad_incomde depends on the lan_id
        position = np.array(crossroad_income[int(index * len(crossroad_income))])
        orientation =utils.orientation_by_position(lanelet,position)

        yaw_rate = 0.0
        slip_angle = 0.0

        return State(velocity=velocity, orientation=orientation, time_step=0, position=position,acceleration=acceleration, yaw_rate=yaw_rate,
                     slip_angle=slip_angle)

    # return a rectangle positioned at the end of a lanelet
    def get_possible_goal_for_lanelet(self, lanelet: Lanelet) -> Rectangle:
        # reverse center vertices to get the distance from lanelet end with cut()
        reversed_center_vertices = LineString(lanelet.center_vertices[::-1])
        goal_center_vertices: LineString = cut(
            reversed_center_vertices, self.config.dist_to_end)[0]
        goal_center = goal_center_vertices.centroid

        # last_point is at index 0 because vertices were reversed previously
        last_point = goal_center_vertices.coords[0]
        orientation = get_orientation_by_coords(
            (goal_center.x, goal_center.y), (last_point[0], last_point[1]))
        return Rectangle(self.config.length, self.config.width, np.array([goal_center.x, goal_center.y]), orientation)

    def get_reachable_lanelets(self, id: int, lanelet_network: LaneletNetwork, planning_problem: PlanningProblem):
        route_planner = RoutePlanner(self.scenario, planning_problem)
        paths = route_planner.find_all_shortest_paths()
        # this will be good to check if init+goal are valid

    def generate_goal(self, goal_area: Rectangle):
        goal_state_list = [
            State(position=goal_area, time_step=Interval(0, self.config.max_time_step))]
        return GoalRegion(goal_state_list)


def get_orientation_by_coords(first_point: Tuple[float, float], next_point: Tuple[float, float]):
    a_x, a_y = next_point
    b_x, b_y = first_point
    # compute orientation: https://stackoverflow.com/questions/42258637/how-to-know-the-angle-between-two-vectors
    return math.atan2(a_y - b_y, a_x - b_x)
