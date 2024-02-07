from FileReader import *
from SolutionPlotter import *
from RouteInitializer import *
from RouteGenerator import *
from Destroy import *
from Repair import *

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

SEED = 1234

vrp_file_path = r'C:\Users\junsick\Desktop\Multi_Modal-main\examples\data\multi_modal_data.vrp'
sol_file_path = r'C:\Users\junsick\Desktop\Multi_Modal-main\examples\data\multi_modal_data.sol'

file_reader = FileReader()
data = file_reader.read_vrp_file(vrp_file_path)
bks = file_reader.read_sol_file(sol_file_path)

### SolutionPlotter 클래스 instance
plotter = SolutionPlotter(data)
"""
plotter.plot_solution(bks)
"""

### RouteInitializer 클래스 instance 
initializer = RouteInitializer(data, k=2, l=1, max_drone_mission=4)
initial_solution = initializer.nearest_neighbor_init_truck()
current_route = initializer.makemakemake(initial_solution)

### currnet route 정보 출력 debugging code
print("\nCurrent routes :", current_route)
print("\nCurrent Objective cost :",MultiModalState(current_route).objective())

### route 플러팅 시각화 debugging code
plotter.plot_current_solution(initial_solution,name="Init Solution(NN/Truck)")
plotter.plot_current_solution(current_route,name="Multi_Modal Solution")


### Destroy 클래스 instance
destroyer = Destroy(current_route)

### Destroy 클래스 debugging code
destroyed_route = destroyer.random_removal(rnd_state)
print("\nAfter random removal:", destroyed_route)

### Destroy 플러팅 시각화 debugging code
plotter.plot_current_solution(destroyed_route,name="Random Removal")

### Repair 클래스 instance
Rep = Repair()

### Repair 클래스 debugging code
repaired_route = Rep.greedy_truck_random_repair(destroyed_route, rnd_state)
print("\nRepaired routes: ", repaired_route['one_path'])

### Repair 플러팅 시각화 debugging code
plotter.plot_current_solution(repaired_route,name="Random Repair")

"""
class Feasibility:
    
    #heuristics/ALNS part 에서 우리가 설정한 제약조건을 만족하는지 checking하는 클래스
    #return 형식 : Ture/False
    
    def function():
        return True,False
   

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
