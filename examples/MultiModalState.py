import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import List

from FileReader import *

vrp_file_path = r'C:\Users\User\OneDrive\바탕 화면\ALNS-master\ALNS-master\examples\data\multi_modal_data.vrp'
sol_file_path = r'C:\Users\User\OneDrive\바탕 화면\ALNS-master\ALNS-master\examples\data\multi_modal_data.sol'

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)
bks = file_reader.read_sol_file(sol_file_path)


class MultiModalState:
    """
    routes 딕셔너리 집합을 input으로 받아서 copy를 수행한 뒤, 해당 routes 에서의 정보를 추출하는 함수
    output: objective cost value / 특정 customer node를 포함한 route  
    """

    def __init__(self, routes):
        self.routes = routes

    def copy(self):
        return MultiModalState(
            copy.deepcopy(self.routes)
        )

    def objective(self):
        """
        data와 routes 딕셔너리 집합을 이용하여 objective value 계산해주는 함수
        our objective cost value = energy_consunmption(kwh)
        energy_consunmption(kwh)={Truck edge cost(km), Truck energy consumption(kwh/km), Drone edge cost(km), Drone energy consumption(kwh/km)}
        TO DO: 이후에 logistic_load 등의 데이터 등을 추가로 활용하여 energy_consumption 모델링 확장 필요
        """
        energy_consumption = 0.0

        for route in self.routes['route']: 
            vtype = route['vtype']
            path = route['path']

            if vtype == 'truck':
                for i in range(len(path) - 1):
                    loc_from = path[i][0] if isinstance(path[i], tuple) else path[i]
                    loc_to = path[i+1][0] if isinstance(path[i+1], tuple) else path[i+1]

                    edge_weight = data["edge_km_t"][loc_from][loc_to]
                    energy_consumption += edge_weight * data["energy_kwh/km_t"]

 
            elif vtype == 'drone': #드론은 1(fly)부터 3(catch)까지만의 edge를 반복적으로 고려해준다는 알고리즘
                start_index = None
                for j in range(len(path)):
                    if path[j][1] == 1:
                        start_index = j
                    elif path[j][1] == 3 and start_index is not None:
                        for k in range(start_index, j):
                            edge_weight = data["edge_km_d"][path[k][0]][path[k+1][0]]
                            energy_consumption += edge_weight * data["energy_kwh/km_d"]
                        start_index = None
 
        return energy_consumption

    
    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()
    

    def find_route(self, customer):
       
        for route in self.routes['route']:
            if customer in route['path']:
                return route
            
        raise ValueError(f"Solution does not contain customer {customer}.")
    