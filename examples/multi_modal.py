import copy
import random
from types import SimpleNamespace
import vrplib 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

SEED = 1234

import matplotlib.pyplot as plt
import numpy as np

from typing import List

def read_vrp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {
        "name": None,
        "type": None,
        "vehicles": None,
        "dimension": None,
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

    section = None
    for line in lines:
        parts = line.split()

        if not parts:
            continue

        keyword = parts[0]
        value = " ".join(parts[1:]).strip()

        if keyword in ["EDGE_KM_D_TYPE:", "EDGE_KM_T_TYPE:"]:
            # Handle edge type separately
            if keyword == "EDGE_KM_D_TYPE:":
                data["edge_km_d_type"] = value
            elif keyword == "EDGE_KM_T_TYPE:":
                data["edge_km_t_type"] = value
            continue

        if section and keyword != section:

            parse_section_data(data, section, line)

        if keyword == "EOF":
            break

        elif keyword == "NAME:":
            data["name"] = value
        elif keyword == "TYPE:":
            data["type"] = value
        elif keyword == "VEHICLES:":
            data["vehicles"] = int(value)
        elif keyword == "DIMENSION:":
            data["dimension"] = int(value)
        elif keyword == "MAXIMUM_SYSTEM_DURATION:":
            data["maximum_system_duration"] = int(value)
        elif keyword == "SERVICETIME:":
            data["service_time"] = int(value)
        elif keyword == "MAX_WAITING_TIME:":
            data["max_waiting_time"] = int(value)
        elif keyword == "INIT_SOC:":
            data["init_soc"] = float(value)
        elif keyword == "MAX_SOC:":
            data["max_soc"] = float(value)
        elif keyword == "MIN_SOC_T:":
            data["min_soc_t"] = float(value)
        elif keyword == "MIN_SOC_D:":
            data["min_soc_d"] = float(value)
        elif keyword == "CAPACITY_T:":
            data["capacity_t"] = float(value)
        elif keyword == "CAPACITY_D:":
            data["capacity_d"] = float(value)
        elif keyword == "CARGO_LIMIT_DRONE:":
            data["cargo_limit_drone"] = float(value)
        elif keyword == "BATTERY_KWH_T:":
            data["battery_kwh_t"] = float(value)
        elif keyword == "BATTERY_KWH_D:":
            data["battery_kwh_d"] = float(value)
        elif keyword == "ENERGY_KWH/KM_T:":
            data["energy_kwh/km_t"] = float(value)
        elif keyword == "ENERGY_KWH/KM_D:":
            data["energy_kwh/km_d"] = float(value)
        elif keyword == "LOSISTIC_KWH/KG_T:":
            data["losistic_kwh/kg_t"] = float(value)
        elif keyword == "LOSISTIC_KWH/KG_D:":
            data["losistic_kwh/kg_d"] = float(value)
        elif keyword == "SPEED_T:":
            data["speed_t"] = float(value)
        elif keyword == "SPEED_D:":
            data["speed_d"] = float(value)
        elif keyword == "NODE_COORD_SECTION":
            section = "NODE_COORD_SECTION"
        elif keyword == "DEMAND_SECTION":
            section = "DEMAND_SECTION"
        elif keyword == "LOGISTIC_LOAD_SECTION":
            section = "LOGISTIC_LOAD_SECTION"
        elif keyword == "AVAILABILITY_LANDING_SPOT_SECTION":
            section = "AVAILABILITY_LANDING_SPOT_SECTION"
        elif keyword == "CUSTOMER_DRONE_PREFERENCE_SECTION":
            section = "CUSTOMER_DRONE_PREFERENCE_SECTION"
        elif keyword == "DEPOT_SECTION":
            section = "DEPOT_SECTION"
        elif keyword == "EDGE_KM_D_FORMAT":
            data["edge_km_d_format"] = value
        elif keyword == "EDGE_KM_T_FORMAT":
            data["edge_km_t_format"] = value
        elif keyword == "EDGE_KM_D":
            section = "EDGE_KM_D"
            data["edge_km_d"] = []
        elif keyword == "EDGE_KM_T":
            section = "EDGE_KM_T"
            data["edge_km_t"] = []

    return data

def parse_section_data(data, section, line):
    parts = line.split()

    if not parts or parts[0] == "EOF":
        return

    if section == "NODE_COORD_SECTION":
        try:
            node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            data["node_coord"][node_id] = (x, y)
        except (ValueError, IndexError):
            pass
    elif section == "DEMAND_SECTION":
        try:
            customer_id, demand = int(parts[0]), int(parts[1])
            data["demand"][customer_id] = demand
        except (ValueError, IndexError):
            pass
    elif section == "LOGISTIC_LOAD_SECTION":
        try:
            customer_id, load = int(parts[0]), int(parts[1])
            data["logistic_load"][customer_id] = load
        except (ValueError, IndexError):
            pass
    elif section == "AVAILABILITY_LANDING_SPOT_SECTION":
        try:
            spot_id, availability = int(parts[0]), int(parts[1])
            data["availability_landing_spot"][spot_id] = availability
        except (ValueError, IndexError):
            pass
    elif section == "CUSTOMER_DRONE_PREFERENCE_SECTION":
        try:
            customer_id, preference = int(parts[0]), int(parts[1])
            data["customer_drone_preference"][customer_id] = preference
        except (ValueError, IndexError):
            pass
    elif section == "DEPOT_SECTION":
        try:
            data["depot"] = int(parts[0])
        except (ValueError, IndexError):
            pass
    elif section == "EDGE_KM_D":
        try:
            data["edge_km_d"].append(list(map(float, parts)))
        except (ValueError, IndexError):
            pass
    elif section == "EDGE_KM_T":
        try:
            data["edge_km_t"].append(list(map(float, parts)))
        except (ValueError, IndexError):
            pass


vrp_file_path = r'C:\Users\User\OneDrive\바탕 화면\ALNS-master\ALNS-master\examples\data\multi_modal_data.vrp'
sol_file_path = r'C:\Users\User\OneDrive\바탕 화면\ALNS-master\ALNS-master\examples\data\multi_modal_data.sol'

def read_sol_file(file_path): #sol_read 함수
    with open(file_path, 'r') as file:
        lines = file.readlines()

    solution = {"routes": [], "cost":None, "vehicle_types":[]}
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


def plot_solution(data, solution, name="VRP Solution"): #plot 함수
    """
    Plot the routes of the passed-in solution.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap('rainbow')

    for route in solution["routes"]:
        ax.plot(
            [data["node_coord"][loc][0] for loc in route],
            [data["node_coord"][loc][1] for loc in route],
            color=cmap(np.random.rand()),  # Random color for each route
            marker='.'
        )

    # Plot the depot
    kwargs = dict(label="Depot", zorder=3, marker="*", s=750)
    ax.scatter(*data["node_coord"][data["depot"]], c="tab:red", **kwargs)

    ax.set_title(f"{name}\n Total distance: {solution['cost']}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)

data = read_vrp_file(vrp_file_path)
print(data)
bks = read_sol_file(sol_file_path)
print(bks)

plot_solution(data, bks, name="Multi_Modal Solution")
plt.show()


class RouteManager:
    def __init__(self, route, k, l, max_drone_mission):
        self.route = route
        self.depot_end = len(route) - 1
        self.can_fly = len(route) - k - l
        self.max_drone_mission = max_drone_mission
        self.k = k
        self.l = l
        self.FLY = 0
        self.SERVICE = 0
        self.CATCH = 0
        self.only_drone_index = []
        self.fly_node_index = []
        self.catch_node_index = []
        self.subroutes = []
        self.generate_subroutes()
 
    def generate_subroutes(self):
        while len(self.subroutes) < self.max_drone_mission:
            self.FLY = random.choice(range(self.CATCH, len(self.route)))
            self.SERVICE = self.FLY + self.k
            self.CATCH = self.SERVICE + self.l
            if self.CATCH > self.depot_end:
                break
            subroute = list(range(self.FLY, self.CATCH + 1))
            self.subroutes.append(subroute)
            self.fly_node_index.append(self.FLY)
            self.only_drone_index.append(self.SERVICE)
            self.catch_node_index.append(self.CATCH)
 
    def get_visit_type(self):
        visit_type = [0] * len(self.route)
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
        return visit_type
 
    def dividing_route(self):
        my_type = self.get_visit_type()
        only_truck = [index for index, value in enumerate(my_type) if value == 4]
        only_drone = [index for index, value in enumerate(my_type) if value == 2]
 
        truck_route = [value for index, value in enumerate(self.route) if index not in only_drone]
        drone_route = [value for index, value in enumerate(self.route) if index not in only_truck]
 
        return truck_route, drone_route
 
route = [0, 2, 3, 5, 8, 7, 6, 1, 4, 0]
route_manager = RouteManager(route, 2, 1, 4)
truck_route, drone_route = route_manager.dividing_route()

print(truck_route,drone_route)

print("박연서 ㅄ")
print(route_manager.get_visit_type())

routes = {
    'num_t' : 1,
    'num_d' : 1,
    'route': [
        {'vtype': 'drone', 'vid': 'd1', 'path': drone_route},
        {'vtype': 'truck', 'vid': 't1', 'path': truck_route}
    ]
}

### 준영아. 드론 라우트를 뽑았을 때 노드 정보 리스트도 함께 데리고 오게 해서 4가 없어진 정보를 기억할 수 있도록 해봐

class MultiModalState:

    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []

    def copy(self):
        return MultiModalState(
            copy.deepcopy(self.routes),
            unassigned=self.unassigned.copy()
        )

    def objective(self, data):
       
        energy_consumption = 0.0

        for route in self.routes.values():
            vtype = route['vtype']
            path = route['path']

            if vtype == 'truck':
                for i in range(len(path) - 1): #트럭은 처음부터 마지막까지 전체엣지를 고려해준다는 알고리즘
                    edge_weight = data["edge_km_t"][path[i]][path[i+1]]
                    energy_consumption += edge_weight * data["energy_kwh/km_t"]

            elif vtype == 'drone':
                for j in range(len(path) - 1):
                    if path[j][1] == 1: #노드에 저장된 정보가 1이면 다음, 다다음 까지의 엣지만 고려해준다는 알고리즘
                        edge_weight = data["edge_km_d"][path[j]][path[j+1]]
                        energy_consumption += edge_weight * data["energy_kwh/km_d"]

                        edge_weight_next = data["edge_km_d"][path[j+1]][path[j+2]]
                        energy_consumption += edge_weight_next * data["energy_kwh/km_d"]

        return energy_consumption
    
    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()
    

    def find_route(self, customer):
       
        for route in self.routes.values():
            if customer in route['path']:
                return route
            
        raise ValueError(f"Solution does not contain customer {customer}.")
    

my_state = MultiModalState(routes)
result = my_state.objective(data)
print(result)
#츌력되는거 확인필요

        
degree_of_destruction = 0.05
customers_to_remove = int((data["dimension"] - 1) * degree_of_destruction)

def random_removal(state, rnd_state):
    """
    Removes a number of randomly selected customers from the passed-in solution.
    """
    destroyed = state.copy()

    for customer in rnd_state.choice(
        range(1, data["dimension"]), customers_to_remove, replace=False
    ):
        destroyed.unassigned.append(customer)
        route = destroyed.find_route(customer)
        route.remove(customer)

    return remove_empty_routes(destroyed)


def remove_empty_routes(state):
    """
    Remove empty routes after applying the destroy operator.
    """
    state.routes = [route for route in state.routes if len(route) != 0]
    return state


def greedy_repair(state, rnd_state):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created.
    """
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
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    """
    best_cost, best_route, best_idx = None, None, None

    for route in state.routes:
        for idx in range(len(route) + 1):

            if can_insert(customer, route):
                cost = insert_cost(customer, route, idx)

                if best_cost is None or cost < best_cost:
                    best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx


def can_insert(customer, route):
    """
    Checks if inserting customer does not exceed vehicle capacity.
    """
    total = data["demand"][route].sum() + data["demand"][customer]
    return total <= data["capacity"]


def insert_cost(customer, route, idx):
    """
    Computes the insertion cost for inserting customer in route at idx.
    """
    dist = data["edge_weight"]
    pred = 0 if idx == 0 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]

    # Increase in cost of adding customer, minus cost of removing old edge
    return dist[pred][customer] + dist[customer][succ] - dist[pred][succ]


def neighbors(customer):
    """
    Return the nearest neighbors of the customer, excluding the depot.
    """
    locations = np.argsort(data["edge_weight"][customer])
    return locations[locations != 0]


def nearest_neighbor():
    """
    Build a solution by iteratively constructing routes, where the nearest
    customer is added until the route has met the vehicle capacity limit.
    """
    routes = []
    unvisited = set(range(1, data["dimension"]))

    while unvisited:
        route = [0]  # Start at the depot
        route_demands = 0

        while unvisited:
            # Add the nearest unvisited customer to the route till max capacity
            current = route[-1]
            nearest = [nb for nb in neighbors(current) if nb in unvisited][0]

            if route_demands + data["demand"][nearest] > data["capacity"]:
                break

            route.append(nearest)
            unvisited.remove(nearest)
            route_demands += data["demand"][nearest]

        customers = route[1:]  # Remove the depot
        routes.append(customers)

    return MultiModalState(routes)

plot_solution(nearest_neighbor(), 'Nearest neighbor solution')


def main():
    alns = ALNS(rnd.RandomState(SEED))
    alns.add_destroy_operator(random_removal)
    alns.add_repair_operator(greedy_repair)
    
    init = nearest_neighbor()
    select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
    accept = RecordToRecordTravel.autofit(init.objective(), 0.02, 0, 9000)
    stop = MaxRuntime(60)

    result = alns.iterate(init, select, accept, stop)

    solution = result.best_state
    objective = solution.objective()
    pct_diff = 100 * (objective - bks.cost) / bks.cost
    
    print(f"Best heuristic objective is {objective}.")
    print(f"This is {pct_diff:.1f}% worse than the optimal solution, which is {bks.cost}.")

    _, ax = plt.subplots(figsize=(12, 6))
    result.plot_objectives(ax=ax)

    
if __name__ == "__main__":
    main()
    plot_solution(nearest_neighbor(), 'Nearest neighbor solution')
    plot_solution(bks, name="Best known solution")
    
    plt.show()