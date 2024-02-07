import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import List

from RouteInitializer import *
from FileReader import *

vrp_file_path = r'C:\Users\82102\Desktop\ALNS-master\examples\data\multi_modal_data.vrp'
sol_file_path = r'C:\Users\82102\Desktop\ALNS-master\examples\data\multi_modal_data.sol'

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)
bks = file_reader.read_sol_file(sol_file_path)

globals
IDLE = 0
FLY = 1
ONLY_DRONE = 2
CATCH = 3
ONLY_TRUCK = 4
UNASSIGNED = 5

initializer = RouteInitializer(data, k=2, l=1, max_drone_mission=4)
initial_solution = initializer.nearest_neighbor_init_truck()

degree_of_destruction = 0.3
customers_to_remove = int((data["dimension"] - 1) * degree_of_destruction)
rnd_state = np.random.RandomState(None)

class Destroy:
    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []
    
    def copy(self):
        return Destroy(
                copy.deepcopy(self.routes),
                unassigned=self.unassigned.copy()
            )

    def random_removal(self, rnd_state):
        """
        내가 설정한 파괴 상수에 따라 파괴할 고객의 수가 결정되고, 그에 따라 랜덤으로 고객노드를 제거한다.

        드론의 path일 경우, 
        1. 파괴 대상 노드가 fly(1) 노드에 해당하면 그에 종속된 service(2) 노드와 함께 제거함과 동시에 그에 종속된 catch(3) 노드는 단순 idle(0) 역할로 바뀐다.
        2. 파괴 대상 노드가 catch(3) 노드에 해당하면 그에 종속된 service(2) 노드와 함께 제거함과 동시에 그에 종속된 fly(1) 노드는 단순 idle(0) 역할로 바뀐다.
        3. 파괴 대상노드가 service(2) 노드에 해당한다면, 그에 종속된 fly(1) 노드와 catch(3) 노드는 모두 단순 idel(0) 역할로 바뀐다.
        4. 파괴 대상노드가 idel(0) 노드에 해당한다면, 해당 노드만 제거를 수행한다.
        주의 : 노드가 캐치가 되자마자, 플라이가 되는 노드일 경우에는 단순 idel(0) 역할이 아니라, 다른 드론의 서브경로의 fly/catch 역할로 정의되어야 한다.
        드론 1 2 1 2 3
        트럭 1 4 1 4 3 의 경우 다시 생각

        트럭의 path일 경우,
        1. 파괴 대상 노드가 fly(1) 노드에 해당하면 해당 노드를 제거함과 동시에 그에 종속된 catch(3) 노드는 단순 idle(0) 역할로 바뀐다.
        또한, 그 중간에 only Truck(4) 노드가 있으면 이 또한 단순 idle(0) 역할로 바뀐다.
        2. 파괴 대상 노드가 catch(3) 노드에 해당하면 해당 노드를 제거함과 동시에 그에 종속된 fly(1) 노드는 단순 idle(0) 역할로 바뀐다.
        또한, 그 중간에 only Truck(4) 노드가 있으면 이 또한 단순 idle(0) 역할로 바뀐다.
        3. 파괴 대상 노드가 idel(0) 노드에 해당한다면, 해당 노드만 제거를 수행한다.
        4. 파괴 대상 노드가 only Truck(4) 노드에 해당한다면, 해당 노드만 제거를 수행한다.

        0 : IDLE
        1 : FLY
        2 : ONLY_DRONE
        3 : CATCH
        4 : ONLY_TRUCK
        5 : UNASSIGNED -> 새로추가 !! 'destroy 된 노드 이자, repair 대상 노드로 작용'

        visit_type을 각각의 상황별로 모두 정의해준 다음에, 5 : UNASSIGNED 정보를 바탕으로 각각의 경로에서 노드 제거 및 종속된 다른 노드들의 visit_type 수정, 그리고 unassigned 튜플리스트 생성
        이를 바탕으로 repair 메소드 수행
        
        truck과 drone이 vid 식별자/ 즉, mother이 같은 것 끼리 pair로 비교해주어야히함

        """
        destroyed = self.copy()
        drone_path = None
        truck_path = None

        for customer in rnd_state.choice(
            range(1, data["dimension"]), customers_to_remove, replace=False
        ):
            
            for route in destroyed.routes['route']:
                if 'vtype' in route and route['vtype'] == 'drone':

                    for i in range(1, len(route["path"])-1):
                         if route['path'][i][0] == customer:
                            if route['path'][i][1] == IDLE:
                                route['path'][i] = (route['path'][i][0], UNASSIGNED)

                    for i in range(1, len(route["path"])-1):
                         if route['path'][i][0] == customer:
                            if route['path'][i][1] == ONLY_DRONE:
                                route['path'][i] = (route['path'][i][0], UNASSIGNED)
                                
                                if route['path'][i+1][1] != UNASSIGNED and route['path'][i+1][1] == CATCH:  #캐치이면서 플라이인 노드 추가 고려
                                    route['path'][i+1] = (route['path'][i+1][0], IDLE)
                                
                                if route['path'][i+1][1] == FLY and route['path'][i+2][1] == ONLY_DRONE: 
                                    route['path'][i+1] = (route['path'][i+1][0], FLY)
                                    
                                if i == 1:
                                    if route['path'][i-1][1] == FLY and route['path'][i-1][1] != UNASSIGNED:
                                        route['path'][i-1] = (route['path'][i-1][0], IDLE)
                                elif i >= 2:
                                    if route['path'][i-1][1] == FLY:
                                        if route['path'][i-2][1] == ONLY_DRONE and route['path'][i-2][1] != UNASSIGNED: #캐치이면서 플라이인 노드 추가 고려
                                            route['path'][i-1] = (route['path'][i-1][0], CATCH)
                                        elif route['path'][i-2][1] != ONLY_DRONE and route['path'][i-2][1] != UNASSIGNED:
                                            route['path'][i-1] = (route['path'][i-1][0], IDLE) 

                    for i in range(1, len(route["path"])-2):
                        if i==1:
                            if route['path'][i][0] == customer:
                                if route['path'][i][1] == FLY:
                                    route['path'][i] = (route['path'][i][0], UNASSIGNED)
                                    route['path'][i+1] = (route['path'][i+1][0], UNASSIGNED)
                                    if route['path'][i+2][1] != UNASSIGNED:
                                        route['path'][i+2] = (route['path'][i+2][0], IDLE)
                                        
                        else:
                            if route['path'][i][0] == customer:
                                if route['path'][i][1] == FLY:
                                    route['path'][i] = (route['path'][i][0], UNASSIGNED)
                                    route['path'][i+1] = (route['path'][i+1][0], UNASSIGNED)
                                    if route['path'][i+2][1] != UNASSIGNED:
                                        route['path'][i+2] = (route['path'][i+2][0], IDLE)
                                        
                                    if route['path'][i-1][1] == ONLY_DRONE and route['path'][i-1][1] != UNASSIGNED: #캐치이면서 플라이인 노드 추가 고려
                                        route['path'][i-1] = (route['path'][i-1][0], UNASSIGNED)
                                        if route['path'][i-2][1] != UNASSIGNED:
                                            route['path'][i-2] = (route['path'][i-2][0], IDLE)
                                    
                    for i in range(2, len(route["path"])-1):
                        if route['path'][i][0] == customer:
                            if route['path'][i][1] == CATCH:
                                route['path'][i] = (route['path'][i][0], UNASSIGNED)
                                route['path'][i-1] = (route['path'][i-1][0], UNASSIGNED)
                                if route['path'][i-2][1] != UNASSIGNED:
                                    route['path'][i-2] = (route['path'][i-2][0], IDLE)

                    for i in range(1, len(route['path']) - 1):
                        if route['path'][i][1] == UNASSIGNED:
                            if (route['path'][i][0], route['path'][i][1]) not in destroyed.unassigned:
                                destroyed.unassigned.append((route['path'][i][0], route['path'][i][1]))
                                
                    drone_path = [point for point in route['path'] if point[1] != UNASSIGNED]


                elif 'vtype' in route and route['vtype'] == 'truck':
                    for i in range(1, len(route["path"]) - 1):
                        if route['path'][i][0] == customer:
                            if route['path'][i][1] == ONLY_TRUCK:
                                route['path'][i] = (route['path'][i][0], 5)
                                if (route['path'][i][0], route['path'][i][1]) not in destroyed.unassigned:
                                    destroyed.unassigned.append((route['path'][i][0], route['path'][i][1]))

                    for i in range(1, len(route["path"]) - 1):
                        if route['path'][i][0] == customer:
                            if route['path'][i][1] == IDLE:
                                route['path'][i] = (route['path'][i][0], UNASSIGNED)
                                del route['path'][i]

                    for i in range(1, len(route["path"]) - 1):
                        if route['path'][i][0] == customer:
                            if route['path'][i][1] == FLY:
                                route['path'][i] = (route['path'][i][0], UNASSIGNED)
                                j = i + 1
                                while j < len(route['path']):
                                    if route['path'][j][1] == ONLY_TRUCK:
                                        if route['path'][j][1] != UNASSIGNED:
                                            route['path'][j] = (route['path'][j][0], 0)
                                    elif route['path'][j][1] == CATCH or route['path'][j][1] == FLY: 
                                        break
                                    j += 1
                                if route['path'][j][1] == CATCH and route['path'][j][1] != UNASSIGNED:
                                    route['path'][j] = (route['path'][j][0], IDLE)
                                elif route['path'][j][1] == 1 and route['path'][j][1] != UNASSIGNED:
                                    route['path'][j] = (route['path'][j][0], FLY)
                                   
                                #캐치이면서 플라이인 노드 추가 고려 
                                k = i - 1
                                while k >= 0:
                                    if route['path'][k][1] == ONLY_TRUCK:
                                        if route['path'][k][1] != UNASSIGNED:
                                            route['path'][k] = (route['path'][k][0], 0)
                                    elif route['path'][k][1] == FLY: 
                                        break
                                    k -= 1
                                if route['path'][k][1] == FLY and route['path'][j][1] != UNASSIGNED:
                                    route['path'][k] = (route['path'][k][0], IDLE)
                                    
                    for i in range(1, len(route["path"]) - 1):
                        if route['path'][i][0] == customer:
                            if route['path'][i][1] == CATCH:
                                route['path'][i] = (route['path'][i][0], UNASSIGNED)
                                j = i - 1
                                while j >= 0 :
                                    if route['path'][j][1] == ONLY_TRUCK:
                                        if route['path'][j][1] != UNASSIGNED:
                                            route['path'][j] = (route['path'][j][0], IDLE)
                                    elif route['path'][j][1] == FLY:
                                        break
                                    j -= 1
                                    break
                                if route['path'][j][1] == FLY and route['path'][j][1] != UNASSIGNED:
                                    route['path'][j] = (route['path'][j][0], IDLE)
                    
                    #드론의 경로와 비교해서 드론의 서비스노드가 제거된 상황을 인식하여 트럭의 VISIT TYPE 업데이트 
                    for i in range(1, len(route['path'])-1):
                        for j in range(1, len(drone_path)-1):
                            if route['path'][i][0] == drone_path[j][0]:
                                route['path'][i] = (drone_path[j][0], drone_path[j][1])
                    i = 0
                    while i < len(route['path']) - 2:
                        if route['path'][i][1] == IDLE and route['path'][i + 1][1] == ONLY_TRUCK:
                            for j in range(i + 1, len(route['path'])):
                                if route['path'][j][1] == IDLE:
                                    break
                                route['path'][j] = (route['path'][j][0], 0)
                            i = j - 1
                        i += 1
           
                    truck_path = [point for point in route['path'] if point[1] != UNASSIGNED]
                  
                    
                if drone_path is not None and truck_path is not None:
                # 트럭의 경로와 드론의 경로를 비교하여 트럭 경로에만 있는 노드가 있으면 드론 경로에 추가
                    for i in range(1, len(truck_path)-1):
                        for j in range(1, len(drone_path)-1):
                            if truck_path[i][0] not in [node[0] for node in drone_path]:
                                if truck_path[i][1] == 0:  # 트럭 경로의 노드가 드론 경로에 없고, visit type이 0인 경우
                                # 트럭 경로의 이전 노드와 동일한 드론 경로의 노드를 찾고, 그 다음 위치에 삽입
                                    if truck_path[i-1][0] == drone_path[j][0]:
                                        drone_path.insert(j+1, truck_path[i]) 
                                        break
                        else:
                            continue
                        break 
                
                if 'vtype' in route and route['vtype'] == 'drone': 
                    route['path'] = drone_path
                    
                elif 'vtype' in route and route['vtype'] == 'truck': 
                    route['path'] = truck_path       
                    
            # nearest_neighbor_truck_path와 동일한 순서로 결과를 출력하기 위해 초기화된 트럭 경로 가져오기
            nearest_neighbor_truck_path = initial_solution['route'][0]['path']

            # 각 경로의 노드 이름을 가져와서 해당 노드가 어디에 속하는지 확인하고 튜플로 변환하여 filled_path 리스트에 추가
            filled_path = []
            for node in nearest_neighbor_truck_path:
                
                for i in range(0, len(destroyed.unassigned)):
                    if node == destroyed.unassigned[i][0]:
                        if node not in [point[0] for point in filled_path]:
                            filled_path.append((node, destroyed.unassigned[i][1]))
                            
                # 노드가 드론 경로에 있는지 확인
                for i in range(0, len(drone_path)-1):
                    if node == drone_path[i][0]:
                        if node not in [point[0] for point in filled_path]:
                            filled_path.append((node, drone_path[i][1]))
                
                for i in range(0, len(truck_path)-1):
                    if node == truck_path[i][0]:
                        if node not in [point[0] for point in filled_path]:
                            filled_path.append((node, truck_path[i][1]))

            # 결과를 nearest_neighbor_truck_path와 동일한 순서로 출력
            result_path = [point for point in filled_path if point[0] in nearest_neighbor_truck_path]
            result_path = result_path[:-1]
            result_path.append(drone_path[-1])
            result_path = [point for point in result_path if point[1] != UNASSIGNED]

        return {'num_t': destroyed.routes['num_t'], 
                'num_d': destroyed.routes['num_d'], 
                'route': destroyed.routes['route'], 
                'unassigned': destroyed.unassigned,
                'one_path':result_path}
