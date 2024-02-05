import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import List

class RouteGenerator:
   
    def __init__(self, state, k, l, max_drone_mission):
        self.routes = state['route']
        # 일단 num_t=1 이라고 가정하고, 한 개의 route만 고려
        # num_t 가 늘어나면, index 돌면서 여러 self.route 저장하여 사용
        self.max_drone_mission = max_drone_mission
        self.k = k
        self.l = l
 
        # self.generate_subroutes()
 
    def makemakemake(self):
        empty_list = []
 
        for route_index, route_info in enumerate(self.routes):
            self.depot_end = len(route_info['path']) - 1
            # self.can_fly = len(self.routes['path']) - k - l
            self.SERVICE = 0
            self.CATCH = 0
            self.only_drone_index = []
            self.fly_node_index = []
            self.catch_node_index = []
            self.subroutes = []
            self.generate_subroutes(route_info['path'])
            diclist = self.dividing_route(self.route_tuples(route_info['path']), route_index)
           
            empty_list.extend(diclist)
 
        return {
            'num_t' : int(len(empty_list)/2),
            'num_d' : int(len(empty_list)/2),
            'route' : empty_list
        }
 
 
    def generate_subroutes(self, each_route):
       
        while len(self.subroutes) < self.max_drone_mission:
            self.FLY = random.choice(range(self.CATCH, len(each_route)))
            self.SERVICE = self.FLY + self.k
            self.CATCH = self.SERVICE + self.l
            if self.CATCH > self.depot_end:
                break
            subroute = list(range(self.FLY, self.CATCH + 1))
            self.subroutes.append(subroute)
            self.fly_node_index.append(self.FLY)
            self.only_drone_index.append(self.SERVICE)
            self.catch_node_index.append(self.CATCH)
 
    def route_tuples(self, each_route):
       
        visit_type = [0] * len(each_route)
        visit_type = [
            1 if index in self.fly_node_index else
            2 if index in self.only_drone_index else
            3 if index in self.catch_node_index else
            0 for index in range(len(visit_type))
        ]
        for i in [
            i for subroute in self.subroutes
            for i in subroute[1:-1] if i not in self.only_drone_index
        ]:
            visit_type[i] = 4
        return list(zip(each_route, visit_type))
 
    def dividing_route(self, route_with_info, route_index):
       
        truck_route = [value for value in route_with_info if value[1] != 2]
        drone_route = [value for value in route_with_info if value[1] != 4]
 
        return [
            {'vtype': 'drone', 'vid': 'd'+ str(route_index + 1), 'path': drone_route},
            {'vtype': 'truck', 'vid': 't'+ str(route_index + 1), 'path': truck_route},
        ]