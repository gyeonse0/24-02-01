from RouteGenerator import *
from FileReader import *

file_reader = FileReader()

vrp_file_path = r'C:\Users\junsick\Desktop\Multi_Modal-main\examples\data\multi_modal_data.vrp'
sol_file_path = r'C:\Users\junsick\Desktop\Multi_Modal-main\examples\data\multi_modal_data.sol'

data = file_reader.read_vrp_file(vrp_file_path)

class Repair():
    """
    1. 원래 state에서 Destroy된 부분을 우선 greedy하게 truck route로 메꾸기 -> (Sacramento, 2019)
    2. 트럭으로 채워서 만든 전체 state에 대하여 random한 sortie를 새로 계산
    """
    
    def greedy_truck_random_repair(self, state, rnd_state):
        """
        Inserts the unassigned customers in the best route.
        If there are no feasible insertions, then a new route is created.
        """

        rnd_state.shuffle(state['unassigned'])

        while len(state['unassigned']) != 0:
            customer = state['unassigned'].pop() # customer는 튜플, (customer_number, 5)형태
            route, idx = self.best_insert(customer, state) # state는 destroyed_route를 받아올 것
            # 어떤 route에 들어가야할지, 그리고 그 route의 몇 번째 index에 들어가야할지를 return
            # 여기서의 route는 destroyed_routes['one_path'] 리스트 안에서 'path'

            if route is not None:
                route.insert(idx, customer) 
                
            else:
                state['one_path'].append([customer])

        state['one_path'] = find_random_sortie(state['one_path'])
        state['route'] = apply_dividing_route_to_routes(state['one_path'])

        return state
    
    def best_insert(self, customer, state):
        """
        Finds the best feasible route and insertion idx for the customer.
        Return (None, None) if no feasible route insertions are found.
        """
        
        best_cost, best_route, best_idx = None, None, None

        for route in state['one_path']: # 여기서의 local route는 'route'리스트 for문 돌아가는 {} 딕셔너리임
                
            for idx in range(len(route) + 1): 

                if self.can_insert(customer, route): 
                    cost = self.insert_cost(customer, route, idx)

                    if best_cost is None or cost < best_cost:
                            best_cost, best_route, best_idx = cost, route, idx

        return best_route, best_idx
    
    def can_insert(self, customer, route):
        """
        이 트럭 route path에 이 customer가 들어올 수 있니?
        일단 현재로서는 feasibility 고려 안한다고 가정
        """

        #total = data["demand"][route].sum() + data["demand"][customer]
        #return total <= data["capacity"]
        return True
    
    def insert_cost(self, customer, route, idx):
        """
        Computes the insertion cost for inserting customer in route at idx.
        """
        customer_num = customer[0]
        dist = data["edge_km_t"]
        pred = 0 if idx == 0 else route[idx - 1][0]
        succ = 0 if idx == len(route) else route[idx][0]

        # Increase in cost of adding customer, minus cost of removing old edge
        return (dist[pred][customer_num] + dist[customer_num][succ] - dist[pred][succ]) * data["energy_kwh/km_d"]
    
