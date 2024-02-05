import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from typing import List

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

SEED = 1234

vrp_file_path = r'C:\Users\User\OneDrive\바탕 화면\ALNS-master\ALNS-master\examples\data\multi_modal_data.vrp'
sol_file_path = r'C:\Users\User\OneDrive\바탕 화면\ALNS-master\ALNS-master\examples\data\multi_modal_data.sol'

class FileReader:
    """
    multi_modal_data.vrp, multi_modal_data.sol 파일을 파싱하고 읽어오는 클래스
    TO DO : 새로운 data 입력 및 수정할 때마다 코드 추가 필요!!
    """
    def __init__(self):
        self.data = {
            "name": None,
            "type": None,
            "vehicles": None,
            "dimension": None,
            "num_t": None,
            "num_d": None,
            "maximum_system_duration": None,
            "service_time": None,
            "max_waiting_time": None,
            "init_soc": None,
            "max_soc": None,
            "min_soc_t": None,
            "min_soc_d": None,
            "capacity_t": None,
            "capacity_d": None,
            "cargo_limit_drone": None,
            "battery_kwh_t": None,
            "battery_kwh_d": None,
            "energy_kwh/km_t": None,
            "energy_kwh/km_d": None,
            "losistic_kwh/kg_t": None,
            "losistic_kwh/kg_d": None,
            "speed_t": None,
            "speed_d": None,
            "node_coord": {},
            "demand": {},
            "logistic_load": {},
            "availability_landing_spot": {},
            "customer_drone_preference": {},
            "depot": None,
            "edge_km_d_type": None,
            "edge_km_t_type": None,
            "edge_km_d_format": None,
            "edge_km_t_format": None,
            "edge_km_d": [],
            "edge_km_t": [],
        }
        self.section = None

    def read_vrp_file(self, file_path):
        """
        multi_modal_data.vrp 파일을 읽어오고, parse_section_data 함수를 이용해서 섹션별로 딕셔너리에 data를 저장해주는 함수
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.split()
            if not parts:
                continue

            keyword = parts[0]
            value = " ".join(parts[1:]).strip()

            if keyword in ["EDGE_KM_D_TYPE:", "EDGE_KM_T_TYPE:"]:
                self.parse_edge_km_type(keyword, value)
                continue
            if self.section and keyword != self.section:
                self.parse_section_data(line)
            if keyword == "EOF":
                break
            elif keyword == "NAME:":
                self.data["name"] = value
            elif keyword == "TYPE:":
                self.data["type"] = value
            elif keyword == "VEHICLES:":
                self.data["vehicles"] = int(value)
            elif keyword == "NUM_T:":
                self.data["num_t"] = int(value)
            elif keyword == "NUM_D:":
                self.data["num_d"] = int(value)
            elif keyword == "DIMENSION:":
                self.data["dimension"] = int(value)
            elif keyword == "MAXIMUM_SYSTEM_DURATION:":
                self.data["maximum_system_duration"] = int(value)
            elif keyword == "SERVICETIME:":
                self.data["service_time"] = int(value)
            elif keyword == "MAX_WAITING_TIME:":
                self.data["max_waiting_time"] = int(value)
            elif keyword == "INIT_SOC:":
                self.data["init_soc"] = float(value)
            elif keyword == "MAX_SOC:":
                self.data["max_soc"] = float(value)
            elif keyword == "MIN_SOC_T:":
                self.data["min_soc_t"] = float(value)
            elif keyword == "MIN_SOC_D:":
                self.data["min_soc_d"] = float(value)
            elif keyword == "CAPACITY_T:":
                self.data["capacity_t"] = float(value)
            elif keyword == "CAPACITY_D:":
                self.data["capacity_d"] = float(value)
            elif keyword == "CARGO_LIMIT_DRONE:":
                self.data["cargo_limit_drone"] = float(value)
            elif keyword == "BATTERY_KWH_T:":
                self.data["battery_kwh_t"] = float(value)
            elif keyword == "BATTERY_KWH_D:":
                self.data["battery_kwh_d"] = float(value)
            elif keyword == "ENERGY_KWH/KM_T:":
                self.data["energy_kwh/km_t"] = float(value)
            elif keyword == "ENERGY_KWH/KM_D:":
                self.data["energy_kwh/km_d"] = float(value)
            elif keyword == "LOSISTIC_KWH/KG_T:":
                self.data["losistic_kwh/kg_t"] = float(value)
            elif keyword == "LOSISTIC_KWH/KG_D:":
                self.data["losistic_kwh/kg_d"] = float(value)
            elif keyword == "SPEED_T:":
                self.data["speed_t"] = float(value)
            elif keyword == "SPEED_D:":
                self.data["speed_d"] = float(value)
            elif keyword == "NODE_COORD_SECTION":
                self.section = "NODE_COORD_SECTION"
            elif keyword == "DEMAND_SECTION":
                self.section = "DEMAND_SECTION"
            elif keyword == "LOGISTIC_LOAD_SECTION":
                self.section = "LOGISTIC_LOAD_SECTION"
            elif keyword == "AVAILABILITY_LANDING_SPOT_SECTION":
                self.section = "AVAILABILITY_LANDING_SPOT_SECTION"
            elif keyword == "CUSTOMER_DRONE_PREFERENCE_SECTION":
                self.section = "CUSTOMER_DRONE_PREFERENCE_SECTION"
            elif keyword == "DEPOT_SECTION":
                self.section = "DEPOT_SECTION"
            elif keyword == "EDGE_KM_D_FORMAT":
                self.data["edge_km_d_format"] = value
            elif keyword == "EDGE_KM_T_FORMAT":
                self.data["edge_km_t_format"] = value
            elif keyword == "EDGE_KM_D":
                self.section = "EDGE_KM_D"
                self.data["edge_km_d"] = []
            elif keyword == "EDGE_KM_T":
                self.section = "EDGE_KM_T"
                self.data["edge_km_t"] = []

        return self.data

    def parse_section_data(self, line):
        """
        multi_modal_data.vrp 파일의 데이터를 섹션별로 알맞게 파싱한 후 data를 저장해주는 함수
        """
        parts = line.split()
        if not parts or parts[0] == "EOF":
            return
        if self.section == "NODE_COORD_SECTION":
            self.parse_node_coord(parts)
        elif self.section == "DEMAND_SECTION":
            self.parse_demand(parts)
        elif self.section == "LOGISTIC_LOAD_SECTION":
            self.parse_logistic_load(parts)
        elif self.section == "AVAILABILITY_LANDING_SPOT_SECTION":
            self.parse_availability_landing_spot(parts)
        elif self.section == "CUSTOMER_DRONE_PREFERENCE_SECTION":
            self.parse_customer_drone_preference(parts)
        elif self.section == "DEPOT_SECTION":
            self.parse_depot(parts)
        elif self.section == "EDGE_KM_D":
            self.parse_edge_km_d(parts)
        elif self.section == "EDGE_KM_T":
            self.parse_edge_km_t(parts)

    def parse_node_coord(self, parts):
        try:
            node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            self.data["node_coord"][node_id] = (x, y)
        except (ValueError, IndexError):
            pass
    def parse_demand(self, parts):
        try:
            customer_id, demand = int(parts[0]), int(parts[1])
            self.data["demand"][customer_id] = demand
        except (ValueError, IndexError):
            pass
    def parse_logistic_load(self, parts):
        try:
            customer_id, load = int(parts[0]), int(parts[1])
            self.data["logistic_load"][customer_id] = load
        except (ValueError, IndexError):
            pass
    def parse_availability_landing_spot(self, parts):
        try:
            spot_id, availability = int(parts[0]), int(parts[1])
            self.data["availability_landing_spot"][spot_id] = availability
        except (ValueError, IndexError):
            pass
    def parse_customer_drone_preference(self, parts):
        try:
            customer_id, preference = int(parts[0]), int(parts[1])
            self.data["customer_drone_preference"][customer_id] = preference
        except (ValueError, IndexError):
            pass
    def parse_depot(self, parts):
        try:
            self.data["depot"] = int(parts[0])
        except (ValueError, IndexError):
            pass
    def parse_edge_km_d(self, parts):
        try:
            self.data["edge_km_d"].append(list(map(float, parts)))
        except (ValueError, IndexError):
            pass
    def parse_edge_km_t(self, parts):
        try:
            self.data["edge_km_t"].append(list(map(float, parts)))
        except (ValueError, IndexError):
            pass
    def parse_edge_km_type(self, keyword, value):
        if keyword == "EDGE_KM_D_TYPE:":
            self.data["edge_km_d_type"] = value
        elif keyword == "EDGE_KM_T_TYPE:":
            self.data["edge_km_t_type"] = value

    def read_sol_file(self, file_path):
        """
        multi_modal_data.sol 파일 읽어와서 파싱 후 데이터 저장해주는 함수
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        solution = {"routes": [], "cost": None, "vehicle_types": []}
        current_route = None

        for line in lines:
            if line.startswith("Route"):
                if current_route is not None:
                    solution["routes"].append(current_route)
                route_parts = line.split(":")[1].strip().split()
                current_route = list(map(int, route_parts))
            elif line.startswith("Cost"):
                solution["cost"] = int(line.split()[1])
            elif line.startswith("Vehicle types"):
                vehicle_types_str = line.split(":")[1].strip()
                vehicle_types = [int(char) for char in vehicle_types_str if char.isdigit()]
                solution["vehicle_types"] = vehicle_types

        if current_route is not None:
            solution["routes"].append(current_route)

        return solution

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)
bks = file_reader.read_sol_file(sol_file_path)

print(data)
"""
print(bks)
plot_solution(data, bks, name="Multi_Modal Solution")
"""

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


plotter = SolutionPlotter(data)
"""
plotter.plot_solution(bks)
"""

class TruckRouteInitializer:
    """
    트럭과 드론의 path 분할에 가장 기초가 되는 트럭만의 route를 NN으로 intialize 하는 클래스
    """
    def __init__(self, data):
        self.data = data

    def neighbors_init_truck(self, customer):
        """
        truck의 distance(km) edge data를 기반으로, 해당 customer의 neighbor 노드 탐색
        """
        locations = np.argsort(self.data["edge_km_t"][customer])
        return locations[locations != 0]
    
    def validate_truck_routes(self,truck_routes):
        """
        모든 트럭의 경로가 한번씩의 주행으로 수요를 만족하는지 검증하는 함수/ 만족하면 pass, 만족하지 않으면 error 발생
        """
        for route in truck_routes:
            consecutive_zeros = sum(1 for loc in route if loc == 0)
            if consecutive_zeros > 2:
                raise ValueError("Unable to satisfy demand with the given number of trucks!!")
            
    def nearest_neighbor_init_truck(self):
        """
        트럭의 capacity 조건을 만족하면서, 가까우면서, 방문한적 없는 노드를 truck_init_route에 순차적으로 append하여 
        truck_init_routes 결정 (num_t로 트럭의 fleet 수 고려)-> 이를 통해 딕셔너리 형태로 route를 저장하고, RouteGenerator의 input route로 적용
        """
        truck_init_routes = [[] for _ in range(self.data["num_t"])]
        unvisited = set(range(1, self.data["dimension"]))

        while unvisited:
            for i in range(self.data["num_t"]):
                route = [0] 
                route_demands = 0

                while unvisited:
                    current = route[-1]
                    neighbors = [nb for nb in self.neighbors_init_truck(current) if nb in unvisited]
                    nearest = neighbors[0]

                    if route_demands + self.data["demand"][nearest] > self.data["capacity_t"]:
                        break

                    route.append(nearest)
                    unvisited.remove(nearest)
                    route_demands += self.data["demand"][nearest]

                route.append(0) 
                truck_init_routes[i].extend(route[0:])
        
        self.validate_truck_routes(truck_init_routes)

        return {
            'num_t': len(truck_init_routes),
            'num_d': 0,
            'route': [{'vtype': 'truck', 'vid': f't{i+1}', 'path': path} for i, path in enumerate(truck_init_routes)]
        }

initializer = TruckRouteInitializer(data)
nearest_routes_t = initializer.nearest_neighbor_init_truck()


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
 
#NN으로 init truck route (nearest_routes_t) 정의 후, routegenerator CLASS 적용
    
route_generator = RouteGenerator(nearest_routes_t, 2, 1, 4)
current_route = route_generator.makemakemake()


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
    

print("\nInit route :", nearest_routes_t)
print("\nCurrent routes :", current_route)
print("\nCurrent Objective cost :",MultiModalState(current_route).objective())

plotter.plot_current_solution(nearest_routes_t,name="Init Solution(NN/Truck)")
plotter.plot_current_solution(current_route,name="Multi_Modal Solution")


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
    
    def __str__(self):
        return str({"routes": self.routes, "unassigned": self.unassigned})
    
    def random_removal(self, rnd_state):
        """
        Removes a number of randomly selected customers from the passed-in solution.
        """
        destroyed = self.copy()

        for customer in rnd_state.choice(
            range(1, data["dimension"]), customers_to_remove, replace=False
        ):
            destroyed.unassigned.append(customer)
            for route in destroyed.routes['route']:
                if 'vtype' in route and route['vtype'] == 'drone':
                    for i, point in enumerate(route['path']):

                        if point[0]==customer and point[1]==1:
                            del route['path'][i]
                            if route['path'][i+1][1] == 2:
                                del route['path'][i+1]

                        elif point[0]==customer and point[1]==3:
                            del route['path'][i]
                            if route['path'][i-1][1] == 2:
                                del route['path'][i-1]

                        elif point[0]==customer and point[1]==2:
                            del route['path'][i]
                        
                        elif point[0] == customer and point[1] == 0:
                            del route['path'][i]
                    

                elif 'vtype' in route and route['vtype'] == 'truck':
                    route['path'] = [point for point in route['path'] if point[0] != customer]

        return destroyed

Debug = Destroy(current_route)
print("destroy input :", Debug.routes)

destroyed_route = Debug.random_removal(rnd_state)
print("Routes after random removal:", destroyed_route)

"""
def greedy_repair(state, rnd_state):
    
    #Inserts the unassigned customers in the best route. If there are no
    #feasible insertions, then a new route is created.

    rnd_state.shuffle(state.unassigned)

    while len(state.unassigned) != 0:
        customer = state.unassigned.pop()
        route, idx = best_insert(customer, state)

        if route is not None:
            route.insert(idx, customer)
        else:
            state.routes.append([customer])

    return state


def best_insert(customer, state):
    
    #Finds the best feasible route and insertion idx for the customer.
    #Return (None, None) if no feasible route insertions are found.
    
    best_cost, best_route, best_idx = None, None, None

    for route in state.routes:
        for idx in range(len(route) + 1):

            if can_insert(customer, route):
                cost = insert_cost(customer, route, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx


def can_insert(customer, route):
    
    #Checks if inserting customer does not exceed vehicle capacity.
    
    total = data["demand"][route].sum() + data["demand"][customer]
    return total <= data["capacity"]


def insert_cost(customer, route, idx):
    
    #Computes the insertion cost for inserting customer in route at idx.
    
    dist = data["edge_weight"]
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]

    # Increase in cost of adding customer, minus cost of removing old edge
    return dist[pred][customer] + dist[customer][succ] - dist[pred][succ]



class Feasibility:
    
    #heuristics/ALNS part 에서 우리가 설정한 제약조건을 만족하는지 checking하는 클래스
    #return 형식 : Ture/False
    
    def function():
        return True,False



heuristics/ALNS part 
   

def main():
    alns = ALNS(rnd.RandomState(SEED))
    alns.add_destroy_operator(random_removal)
    alns.add_repair_operator(greedy_repair)
    
    init = MultiModalState(nearest_routes_t) 
    #our init : 트럭nn으로 정렬
    #수정 전 original : init = nearest_neighbor() 이 함수 return이 class 였음
    select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
    accept = RecordToRecordTravel.autofit(init.objective(), 0.02, 0, 9000)
    stop = MaxRuntime(60)

    result = alns.iterate(init, select, accept, stop)

    solution = result.best_state
    objective = solution.objective()
    pct_diff = 100 * (objective - bks.cost) / bks.cost
    
    print(f"Best heuristic objective is {objective}.")
    print(f"This is {pct_diff:.1f}%  worse than the optimal solution, which is {bks.cost}.")

    _, ax = plt.subplots(figsize=(12, 6))
    result.plot_objectives(ax=ax)

if __name__ == "__main__":
    main()
"""
