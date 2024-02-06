import numpy as np
import random

class RouteInitializer:
    def __init__(self, data, k, l, max_drone_mission):
        self.data = data
        self.k = k
        self.l = l
        self.max_drone_mission = max_drone_mission

    def neighbors_init_truck(self, customer):
        locations = np.argsort(self.data["edge_km_t"][customer])
        return locations[locations != 0]

    def validate_truck_routes(self, truck_routes):
        for route in truck_routes:
            consecutive_zeros = sum(1 for loc in route if loc == 0)
            if consecutive_zeros > 2:
                raise ValueError("Unable to satisfy demand with the given number of trucks!!")

    def nearest_neighbor_init_truck(self):
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
            'route': [{'vtype': 'truck', 'vid': f't{i + 1}', 'path': path} for i, path in enumerate(truck_init_routes)]
        }

    def makemakemake(self, state):
        empty_list = []

        for route_index, route_info in enumerate(state['route']):
            self.depot_end = len(route_info['path']) - 1
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
            'num_t': int(len(empty_list) / 2),
            'num_d': int(len(empty_list) / 2),
            'route': empty_list
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
            {'vtype': 'drone', 'vid': 'd' + str(route_index + 1), 'path': drone_route},
            {'vtype': 'truck', 'vid': 't' + str(route_index + 1), 'path': truck_route},
        ]
