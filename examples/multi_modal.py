import copy
from types import SimpleNamespace
import vrplib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

from FileReader import *
from SolutionPlotter import *
from RouteInitializer import *
from RouteGenerator import *
from Destroy import *
from Repair import *
from MultiModalState import *

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

SEED = 1234
rnd_state = np.random.RandomState(None)


vrp_file_path = r'C:\Users\82102\Desktop\ALNS-master\examples\data\multi_modal_data.vrp'
sol_file_path = r'C:\Users\82102\Desktop\ALNS-master\examples\data\multi_modal_data.sol'

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)
bks = file_reader.read_sol_file(sol_file_path)

### RouteInitializer 클래스 instance 
initializer = RouteInitializer(data, k=2, l=1, max_drone_mission=4)
initial_solution = initializer.nearest_neighbor_init_truck()
initial_truck = initializer.init_truck()

print("\nonly truck's(NN)", initial_truck)


current_route = initializer.makemakemake(initial_solution)
### currnet route 정보 출력 debugging code
print("\nCurrent's", current_route)
print("Current Objective cost :",MultiModalState(current_route).objective())


### SolutionPlotter 클래스 instance
plotter = SolutionPlotter(data)
### route 플러팅 시각화 debugging code
plotter.plot_current_solution(initial_truck,name="Init Solution(NN/Truck)")
plotter.plot_current_solution(current_route,name="Multi_Modal Solution")


### Destroy 클래스 debugging code
destroyer = Destroy()
destroyed_route = destroyer.random_removal(current_route,rnd_state)
print("\nrandom removal's", destroyed_route)


### Destroy 플러팅 시각화 debugging code
plotter.plot_current_solution(destroyed_route,name="Random Removal")

### Repair 클래스 debugging code
Rep = Repair()
repaired_route = Rep.greedy_truck_random_repair(destroyed_route,rnd_state)
print("\nRepair's", repaired_route)

### Repair 플러팅 시각화 debugging code
plotter.plot_current_solution(repaired_route,name="Random Repair")

"""
class Feasibility:
    
    #heuristics/ALNS part 에서 우리가 설정한 제약조건을 만족하는지 checking하는 클래스
    #return 형식 : Ture/False
    
    def function():
        return True,False


alns = ALNS(rnd.RandomState(None))
alns.add_destroy_operator(destroyer.random_removal)
alns.add_repair_operator(Rep.greedy_truck_random_repair)

init = initializer.makemakemake(initial_solution)

select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
accept = RecordToRecordTravel.autofit(init.objective(), 0.02, 0, 5000)
stop = MaxRuntime(60)

result = alns.iterate(init, select, accept, stop)
print("ALNS Result:", result)

solution = result.best_state
objective = solution.objective()
pct_diff = 100 * (objective - 10) / 10

print(f"Best heuristic objective is {objective}.")
print(f"This is {pct_diff:.1f}%  worse than the optimal solution, which is .")

_, ax = plt.subplots(figsize=(12, 6))
result.plot_objectives(ax=ax)
plt.show()
"""
