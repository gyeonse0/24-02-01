import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import List

from MultiModalState import *

class SolutionPlotter:
    """
    특정 route를 기반으로 location 및 path, cost 정보등을 시각화 해주는 클래스
    """
    def __init__(self, data):
        self.data = data

    def plot_solution(self, solution, name="Multi_Modal Solution(bks)"):
        """
        vrp, sol 파일 기반으로 plot 해주는 함수 
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.get_cmap('rainbow')

        for route in solution["routes"]:
            ax.plot(
                [self.data["node_coord"][loc][0] for loc in route],
                [self.data["node_coord"][loc][1] for loc in route],
                color=cmap(np.random.rand()),  
                marker='.'
            )

        kwargs = dict(label="Depot", zorder=3, marker="*", s=750)
        ax.scatter(*self.data["node_coord"][self.data["depot"]], c="tab:red", **kwargs)
        for node, (x, y) in self.data["node_coord"].items():
            ax.annotate(str(node), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
        ax.set_title(f"{name}\nTotal Energy Consumption(cost): {solution['cost']} kwh")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.legend(frameon=False, ncol=3)
        plt.show()

    def plot_current_solution(self, routes, name="Multi_Modal Solution"):
        """
        우리가 뽑아낸 routes 딕셔너리 집합과 solution class를 통해서 현재의 cost와 path를 plot 해주는 함수
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.get_cmap('rainbow')

        for route_info in routes['route']:
            vtype = route_info['vtype']
            vid = route_info['vid']
            path = route_info['path']

            if vtype == 'drone':
                color = 'b'
                path = path if isinstance(path, list) else path[0]
                loc_getter = lambda loc: loc[0] if isinstance(loc, tuple) else loc

            elif vtype == 'truck':
                path = path if isinstance(path, list) else path[0]
                color = 'g'
                loc_getter = lambda loc: loc[0] if isinstance(loc, tuple) else loc

            else:
                color = 'k'
                loc_getter = lambda loc: loc

            ax.plot(
                [self.data['node_coord'][loc_getter(loc)][0] for loc in path],
                [self.data['node_coord'][loc_getter(loc)][1] for loc in path],
                color=color,
                marker='.',
                label=f'{vtype} {vid}'
            )



        kwargs = dict(label="Depot", zorder=3, marker="*", s=750)
        ax.scatter(*self.data["node_coord"][self.data["depot"]], c="tab:red", **kwargs)
        for node, (x, y) in self.data["node_coord"].items():
            ax.annotate(str(node), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
        ax.set_title(f"{name}\nTotal Energy Consumption(cost): {MultiModalState(routes).objective()} kWh")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.legend(frameon=False, ncol=3)
        plt.show()


        kwargs = dict(label="Depot", zorder=3, marker="*", s=750)
        ax.scatter(*self.data["node_coord"][self.data["depot"]], c="tab:red", **kwargs)
        for node, (x, y) in self.data["node_coord"].items():
            ax.annotate(str(node), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
        ax.set_title(f"{name}\nTotal Energy Consumption(cost): {MultiModalState(routes).objective()} kWh")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.legend(frameon=False, ncol=3)
        plt.show()