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
                            if route['path'][i+1][1] == 2:
                                del route['path'][i:i+2]

                        elif point[0]==customer and point[1]==3:
                            if route['path'][i-1][1] == 2:
                                del route['path'][i-1:i+1]

                        elif point[0]==customer and point[1]==2:
                            del route['path'][i]
                        
                        elif point[0] == customer and point[1] == 0:
                            del route['path'][i]
                    

                elif 'vtype' in route and route['vtype'] == 'truck':
                    route['path'] = [point for point in route['path'] if point[0] != customer]

        return destroyed