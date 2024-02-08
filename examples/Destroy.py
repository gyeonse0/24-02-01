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

vrp_file_path = r'C:\Users\User\OneDrive\바탕 화면\ALNS-master\ALNS-master\examples\data\multi_modal_data.vrp'
sol_file_path = r'C:\Users\User\OneDrive\바탕 화면\ALNS-master\ALNS-master\examples\data\multi_modal_data.sol'

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)
bks = file_reader.read_sol_file(sol_file_path)

globals
IDLE = 0 #코멘트 추가필요
FLY = 1
ONLY_DRONE = 2
CATCH = 3
ONLY_TRUCK = 4
NULL = None 

initializer = RouteInitializer(data, k=2, l=1, max_drone_mission=4)
initial_solution = initializer.nearest_neighbor_init_truck()

degree_of_destruction = 0.3
customers_to_remove = int((data["dimension"] - 1) * degree_of_destruction)
rnd_state = np.random.RandomState(None)

class Destroy:
    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []
    
    def copy(self): # _copy 숨겨진 FUNCTION 주의
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
        드론 1 2 1 2 3 .... 중간에 캐치되자마자 플라이가 필요한가 ... (새로운 인자로 설정고려)
        트럭 1 4 1 4 3 의 경우 다시 생각

        트럭의 path일 경우,
        1. 파괴 대상 노드가 fly(1) 노드에 해당하면 해당 노드를 제거함과 동시에 그에 종속된 catch(3) 노드는 단순 idel(0) 역할로 바뀐다.
        또한, 그 중간에 only Truck(4) 노드가 있으면 이 또한 단순 idel(0) 역할로 바뀐다.
        2. 파괴 대상 노드가 catch(3) 노드에 해당하면 해당 노드를 제거함과 동시에 그에 종속된 fly(1) 노드는 단순 idel(0) 역할로 바뀐다.
        또한, 그 중간에 only Truck(4) 노드가 있으면 이 또한 단순 idel(0) 역할로 바뀐다.
        3. 파괴 대상 노드가 idel(0) 노드에 해당한다면, 해당 노드만 제거를 수행한다.
        4. 파괴 대상 노드가 only Truck(4) 노드에 해당한다면, 해당 노드만 제거를 수행한다.

        0 : IDLE
        1 : FLY
        2 : ONLY_DRONE
        3 : CATCH
        4 : ONLY_TRUCK
        NULL : NULL -> 새로추가 !! 'destroy 된 노드 이자, repair 대상 노드로 작용'

        visit_type을 각각의 상황별로 모두 정의해준 다음에, 5 : UNASSIGNED 정보를 바탕으로 각각의 경로에서 노드 제거 및 종속된 다른 노드들의 visit_type 수정, 그리고 unassigned 튜플리스트 생성
        이를 바탕으로 repair 메소드 수행
        
        truck과 drone이 vid 식별자/ 즉, mother이 같은 것 끼리 pair로 비교해주어야히함

        DESTROY에서 다 NULL(초기화)을 만들어주고, REPAIR에서 VISIT_TYPE 업데이트 !!
        
        연구 하다가 드론이 두곳 이상 서비스 가능하다고 열어두고 생각해보기 !!

        오퍼레이터 클래스 -> (디스트로이+리페어 ) + DEF UPDATE

        카카오톡 상황 오류 확인..
        """
        destroyed = self.copy()
        drone_path = None
        truck_path = None

        for customer in rnd_state.choice(
            range(1, data["dimension"]), customers_to_remove, replace=False):
            
            for route in destroyed.routes['route']:
                if 'vtype' in route and route['vtype'] == 'drone':

                    for i in range(1, len(route["path"])-1):
                         if route['path'][i][0] == customer:
                            if route['path'][i][1] == IDLE:
                                route['path'][i] = (route['path'][i][0], NULL)

                    for i in range(1, len(route["path"])-1):
                         if route['path'][i][0] == customer:
                            if route['path'][i][1] == ONLY_DRONE:
                                route['path'][i] = (route['path'][i][0], NULL)
                                
                                if route['path'][i+1][1] != NULL and route['path'][i+1][1] == CATCH:  #캐치이면서 플라이인 노드 추가 고려
                                    route['path'][i+1] = (route['path'][i+1][0], IDLE)
                                
                                if route['path'][i+1][1] == FLY and route['path'][i+2][1] == ONLY_DRONE: 
                                    route['path'][i+1] = (route['path'][i+1][0], FLY)
                                    
                                if i == 1:
                                    if route['path'][i-1][1] == FLY and route['path'][i-1][1] != NULL:
                                        route['path'][i-1] = (route['path'][i-1][0], IDLE)
                                elif i >= 2:
                                    if route['path'][i-1][1] == FLY:
                                        if route['path'][i-2][1] == ONLY_DRONE and route['path'][i-2][1] != NULL: #캐치이면서 플라이인 노드 추가 고려
                                            route['path'][i-1] = (route['path'][i-1][0], CATCH)
                                        elif route['path'][i-2][1] != ONLY_DRONE and route['path'][i-2][1] != NULL:
                                            route['path'][i-1] = (route['path'][i-1][0], IDLE) 

                    for i in range(1, len(route["path"])-2):
                        if i==1:
                            if route['path'][i][0] == customer:
                                if route['path'][i][1] == FLY:
                                    route['path'][i] = (route['path'][i][0], NULL)
                                    route['path'][i+1] = (route['path'][i+1][0], NULL)
                                    if route['path'][i+2][1] != NULL:
                                        route['path'][i+2] = (route['path'][i+2][0], IDLE)
                                        
                        elif i >= 2:
                            if route['path'][i][0] == customer:
                                if route['path'][i][1] == FLY:
                                    route['path'][i] = (route['path'][i][0], NULL)
                                    route['path'][i+1] = (route['path'][i+1][0], NULL)
                                    if route['path'][i+2][0] == CATCH and route['path'][i+2][1] != NULL:
                                        route['path'][i+2] = (route['path'][i+2][0], IDLE)
                                        
                                    if route['path'][i-1][1] == ONLY_DRONE and route['path'][i-1][1] != NULL: #캐치이면서 플라이인 노드 추가 고려
                                        route['path'][i-1] = (route['path'][i-1][0], NULL)
                                        if route['path'][i-2][1] == FLY and route['path'][i-2][1] != NULL:
                                            route['path'][i-2] = (route['path'][i-2][0], IDLE)
                                    
                    for i in range(2, len(route["path"])-1):
                        if route['path'][i][0] == customer:
                            if route['path'][i][1] == CATCH:
                                route['path'][i] = (route['path'][i][0], NULL)
                                route['path'][i-1] = (route['path'][i-1][0], NULL)
                                if route['path'][i-2][1] != NULL:
                                    route['path'][i-2] = (route['path'][i-2][0], IDLE)

                    for i in range(1, len(route['path']) - 1):
                        if route['path'][i][1] == NULL:
                            if (route['path'][i][0], route['path'][i][1]) not in destroyed.unassigned:
                                destroyed.unassigned.append((route['path'][i][0], route['path'][i][1]))
                                
                    route['path'] = [point for point in route['path'] if point[1] != NULL]
                    
                    drone_path=route['path']


                # 트럭의 경우 
                elif 'vtype' in route and route['vtype'] == 'truck':
                    for i in range(1, len(route["path"]) - 1):
                        if route['path'][i][0] == customer:
                            if route['path'][i][1] == ONLY_TRUCK:
                                route['path'][i] = (route['path'][i][0], NULL)
                                if (route['path'][i][0], route['path'][i][1]) not in destroyed.unassigned:
                                    destroyed.unassigned.append((route['path'][i][0], route['path'][i][1]))

                    for i in range(1, len(route["path"]) - 1):
                        if route['path'][i][0] == customer:
                            if route['path'][i][1] == IDLE:
                                route['path'][i] = (route['path'][i][0], NULL)
                                del route['path'][i]

                    for i in range(1, len(route["path"]) - 1):
                        if route['path'][i][0] == customer:
                            if route['path'][i][1] == FLY:
                                route['path'][i] = (route['path'][i][0], NULL)
                                j = i + 1
                                while j < len(route['path']):
                                    if route['path'][j][1] == ONLY_TRUCK:
                                        if route['path'][j][1] != NULL:
                                            route['path'][j] = (route['path'][j][0], 0)
                                    elif route['path'][j][1] == CATCH or route['path'][j][1] == FLY: 
                                        break
                                    j += 1
                                if route['path'][j][1] == CATCH and route['path'][j][1] != NULL:
                                    route['path'][j] = (route['path'][j][0], IDLE)
                                elif route['path'][j][1] == 1 and route['path'][j][1] != NULL:
                                    route['path'][j] = (route['path'][j][0], FLY)
                                   
                                #캐치이면서 플라이인 노드 추가 고려 
                                k = i - 1
                                while k >= 0:
                                    if route['path'][k][1] == ONLY_TRUCK:
                                        if route['path'][k][1] != NULL:
                                            route['path'][k] = (route['path'][k][0], 0)
                                    elif route['path'][k][1] == FLY: 
                                        break
                                    k -= 1
                                if route['path'][k][1] == FLY and route['path'][j][1] != NULL:
                                    route['path'][k] = (route['path'][k][0], IDLE)
                                    
                    for i in range(1, len(route["path"]) - 1):
                        if route['path'][i][0] == customer:
                            if route['path'][i][1] == CATCH:
                                route['path'][i] = (route['path'][i][0], NULL)
                                j = i - 1
                                while j >= 0 :
                                    if route['path'][j][1] == ONLY_TRUCK:
                                        if route['path'][j][1] != NULL:
                                            route['path'][j] = (route['path'][j][0], IDLE)
                                    elif route['path'][j][1] == FLY:
                                        break
                                    j -= 1
                                    break
                                if route['path'][j][1] == FLY and route['path'][j][1] != NULL:
                                    route['path'][j] = (route['path'][j][0], IDLE)
                    
                    #드론의 경로와 비교해서 드론의 서비스노드가 제거된 상황을 인식하여 트럭의 VISIT TYPE 업데이트 
                    
                    if truck_path is not None and drone_path is not None:
                        for i in range(0, len(route['path'])-1):
                            for j in range(1, len(drone_path)-1):
                                if route['path'][i][0] == drone_path[j][0]:
                                    if drone_path[j][1] == IDLE:
                                        route['path'][i] = (drone_path[j][0], drone_path[j][1])

                    for i in range(0, len(route['path'])-1):
                        if route['path'][i][1] == IDLE and route['path'][i + 1][1] == ONLY_TRUCK:
                            route['path'][i+1] = (route['path'][i+1][0], IDLE)

           
                    truck_path = [point for point in route['path'] if point[1] != NULL]
                
                if 'vtype' in route and route['vtype'] == 'drone': 
                    route['path'] = drone_path
                    
                elif 'vtype' in route and route['vtype'] == 'truck': 
                    route['path'] = truck_path      
                    
        one_path = []
        for i in range(len(destroyed.routes['route']) // 2):
            nearest_neighbor_truck_path = initial_solution['route'][i]['path']
            filled_path=[]
            # 드론과 트럭 경로 각각 가져오기
            drone_route = destroyed.routes['route'][2 * i]['path']
            truck_route = destroyed.routes['route'][2 * i + 1]['path']
            
            # 파괴된 경로의 노드 정보 추가
            for unassigned_node, unassigned_value in destroyed.unassigned:
                if unassigned_node not in [point[0] for point in filled_path]:
                    filled_path.append((unassigned_node, unassigned_value))
            
            # 드론 경로의 노드 정보 추가
            for node, value in drone_route[:-1]:
                if node not in [point[0] for point in filled_path]:
                    filled_path.append((node, value))
            
            # 트럭 경로의 노드 정보 추가
            for node, value in truck_route[:-1]:
                if node not in [point[0] for point in filled_path]:
                    filled_path.append((node, value))
            
            # nearest_neighbor_truck_path에 속하지 않은 노드 정보 추가
            for node in nearest_neighbor_truck_path:
                if node not in [point[0] for point in filled_path]:
                    filled_path.append((node, NULL))

            #결과를 nearest_neighbor_truck_path와 동일한 순서로 출력
            filled_path = sorted(filled_path, key=lambda x: nearest_neighbor_truck_path.index(x[0]))
            filled_path.append(drone_route[-1])
            filled_path = [point for point in filled_path if point[1] != NULL]
            one_path.append(filled_path)
            
        return {'num_t': destroyed.routes['num_t'], 
                'num_d': destroyed.routes['num_d'], 
                'route': destroyed.routes['route'], 
                'unassigned': destroyed.unassigned,
                'one_path': one_path} 
