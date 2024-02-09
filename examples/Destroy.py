import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import List

from RouteGenerator import *
from RouteInitializer import *
from FileReader import *

vrp_file_path = r'C:\Users\82102\Desktop\ALNS-master\examples\data\multi_modal_data.vrp'
sol_file_path = r'C:\Users\82102\Desktop\ALNS-master\examples\data\multi_modal_data.sol'

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
        one_path 읽어와서 visit type update 후, 분할하면 훨씬 간단 -> 2, 4 는 1, 3 사이에 올 수 밖에 없음을 이용
        1 2 1 2 3 
        1 4 1 4 3 만 추가고려 
        
        0 : IDLE
        1 : FLY
        2 : ONLY_DRONE
        3 : CATCH
        4 : ONLY_TRUCK
        NULL : NULL -> 새로추가 !! 'destroy 된 노드 이자, repair 대상 노드로 작용'

        DESTROY에서 다 NULL(초기화)을 만들어주고, REPAIR에서 VISIT_TYPE 업데이트 ?
        -> DESTROY 했는데 드론 경로가 살아있어도 다 무시하고 그냥 0 으로 ?
        
        연구 하다가 드론이 두곳 이상 서비스 가능하다고 열어두고 생각해보기 !!

        오퍼레이터 클래스 -> (디스트로이+리페어 ) + DEF UPDATE
        """
        destroyed = self.copy()
        drone_path = None
        truck_path = None

        for customer in rnd_state.choice(
            range(1, data["dimension"]), customers_to_remove, replace=False):
            
            for route in destroyed.routes["one_path"]:
                
                for i in range(0, len(route) - 1):
                    if route[i][0] == customer:
                        if route[i][1] == IDLE:
                            route[i] = (route[i][0], NULL)
                            if (route[i][0], route[i][1]) not in destroyed.unassigned:
                                destroyed.unassigned.append((route[i][0], route[i][1]))
                                
                for i in range(1, len(route)-1):
                    if route[i][0] == customer:
                        if route[i][1] == ONLY_DRONE:
                            route[i] = (route[i][0], NULL)
                            if (route[i][0], route[i][1]) not in destroyed.unassigned:
                                destroyed.unassigned.append((route[i][0], route[i][1]))
                            
                            if i == 1:
                                if route[i - 1][1] == FLY:
                                    route[i - 1] = (route[i - 1][0], IDLE)
                                
                            j = i + 1
                            while j<= len(route) and (route[j][1] != FLY and route[j][1] != CATCH):
                                if route[j][1] == ONLY_DRONE:
                                    route[j] = (route[j][0], NULL)
                                    if (route[j][0], route[j][1]) not in destroyed.unassigned:
                                        destroyed.unassigned.append((route[j][0], route[j][1]))
                                elif route[j][1] == ONLY_TRUCK and route[j][1] != NULL:
                                    route[j] = (route[j][0], IDLE)
                                j += 1
                                
                            if route[j][1] == CATCH:  
                                route[j] = (route[j][0], IDLE)
                            
                            
                            if i >= 2 :
                                k = i - 1
                                while k >= 0 and route[k][1] != FLY:
                                    if route[k][1] == ONLY_DRONE:
                                        route[k] = (route[k][0], NULL)
                                        if (route[k][0], route[k][1]) not in destroyed.unassigned:
                                            destroyed.unassigned.append((route[k][0], route[k][1]))
                                    elif route[k][1] == ONLY_TRUCK:
                                        route[k] = (route[k][0], IDLE)
                                    k -= 1
                                
                                if k == 0 and route[k][1] == FLY:
                                    route[k] = (route[k][0], IDLE)
                                elif k > 0 and route[k][1] == FLY:
                                    if route[k - 1][1] == CATCH:
                                        route[k] = (route[k][0], IDLE)
                                    elif route[k - 1][1] == IDLE:
                                        route[k] = (route[k][0], IDLE)
                                    elif route[k - 1][1] == ONLY_DRONE:
                                        route[k] = (route[k][0], CATCH)
                                    elif route[k - 1][1] == ONLY_TRUCK:
                                        route[k] = (route[k][0], CATCH)

            
                for i in range(2, len(route)-1):
                    if route[i][0] == customer:
                        if route[i][1] == CATCH:
                            route[i] = (route[i][0], NULL)
                            if (route[i][0], route[i][1]) not in destroyed.unassigned:
                                destroyed.unassigned.append((route[i][0], route[i][1]))
                            j = i - 1
                            while j>0 and route[j][1] != FLY:
                                if route[j][1] == ONLY_DRONE:
                                    route[j] = (route[j][0], NULL)
                                    if (route[j][0], route[j][1]) not in destroyed.unassigned:
                                        destroyed.unassigned.append((route[j][0], route[j][1]))
                                elif route[j][1] == ONLY_TRUCK and route[j][1] != NULL:
                                    route[j] = (route[j][0], IDLE)
                                j -= 1
                            
                            if j==0:
                                if route[j][1] == FLY:
                                    route[j] = (route[j][0], IDLE)
                                
                            elif j>=1:
                                if route[j][1] == FLY:
                                    if route[j-1][1] == CATCH and route[j-1][1] != NULL:
                                        route[j] = (route[j][0], IDLE)
                                    elif route[j-1][1] == IDLE and route[j-1][1] != NULL:
                                        route[j] = (route[j][0], IDLE)
                                    elif route[j-1][1] == ONLY_DRONE and route[j-1][1] != NULL:
                                        route[j] = (route[j][0], CATCH)
                                    elif route[j-1][1] == ONLY_TRUCK and route[j-1][1] != NULL:
                                        route[j] = (route[j][0], CATCH)
                            
                for i in range(1,len(route)-1):
                    if route[i][0] == customer:
                        if route[i][1] == ONLY_TRUCK:
                            route[i] = (route[i][0], NULL)
                            if (route[i][0], route[i][1]) not in destroyed.unassigned:
                                destroyed.unassigned.append((route[i][0], route[i][1]))
                                
                                
            destroyed.routes['one_path'] = [[point for point in route if point[1] is not None] for route in destroyed.routes['one_path']]
            divided_routes = apply_dividing_route_to_routes(destroyed.routes['one_path'])

        return {'num_t':len(destroyed.routes["one_path"]),
                'num_d':len(destroyed.routes["one_path"]),
                'one_path': destroyed.routes["one_path"],
                'unassigned': destroyed.unassigned,
                'route': divided_routes}
