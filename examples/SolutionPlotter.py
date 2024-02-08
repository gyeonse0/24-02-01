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

    def plot_current_solution(self, routes, name="Multi_Modal Solution"):
        """
        우리가 뽑아낸 routes 딕셔너리 집합과 solution class를 통해서 현재의 cost와 path를 plot 해주는 함수
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.get_cmap('rainbow')

        used_colors = set()  # 사용된 색상을 추적하기 위한 세트

        for route_info in routes['route']:
            vtype = route_info['vtype']
            vid = route_info['vid']
            path = route_info['path']

            if vtype == 'drone':
                color = self.generate_unique_color(used_colors)  # 겹치지 않는 색상 생성
                used_colors.add(tuple(color))
                path = path if isinstance(path, list) else path[0]
                loc_getter = lambda loc: loc[0] if isinstance(loc, tuple) else loc
                linestyle = '--'

            elif vtype == 'truck':
                color = self.generate_unique_color(used_colors)  # 겹치지 않는 색상 생성
                used_colors.add(tuple(color))
                path = path if isinstance(path, list) else path[0]
                loc_getter = lambda loc: loc[0] if isinstance(loc, tuple) else loc
                linestyle = '-'

            else:
                color = 'k'
                loc_getter = lambda loc: loc

            # 경로 그리기
            ax.plot(
                [self.data['node_coord'][loc_getter(loc)][0] for loc in path],
                [self.data['node_coord'][loc_getter(loc)][1] for loc in path],
                color=color,
                linestyle=linestyle, 
                linewidth=2,
                marker='.',
                label=f'{vtype} {vid}'
            )

            # 방향 화살표 그리기
            for i in range(len(path)-1):
                start = self.data['node_coord'][loc_getter(path[i])]
                end = self.data['node_coord'][loc_getter(path[i+1])]
                ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color))

        kwargs = dict(label="Depot", zorder=3, marker="s", s=80)
        ax.scatter(*self.data["node_coord"][self.data["depot"]], c="tab:red", **kwargs)
        for node, (x, y) in self.data["node_coord"].items():
            ax.annotate(str(node), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')
        ax.set_title(f"{name}\nTotal Energy Consumption(cost): {MultiModalState(routes).objective()} kWh")
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        ax.legend(frameon=False, ncol=3)
        plt.show()

    def generate_unique_color(self, used_colors):
        """
        겹치지 않는 색상을 생성하는 함수
        """
        while True:
            color = [random.random() for _ in range(3)]  # 무작위 RGB 값 생성
            if tuple(color) not in used_colors:
                return color

