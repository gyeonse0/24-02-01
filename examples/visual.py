import matplotlib.pyplot as plt

# 주어진 좌표 데이터
node_coordinates = {
    1: (37.498744, 127.027566),
    2: (37.496011, 127.037548),
    3: (37.491210, 127.058528),
    4: (37.488537, 127.062482),
    5: (37.486450, 127.031767),
    6: (37.488624, 127.025840),
    7: (37.486841, 127.045192),
    8: (37.493832, 127.056170)
}

# 노드 좌표 추출
x_coords, y_coords = zip(*node_coordinates.values())

# 노드 플로팅
plt.scatter(x_coords, y_coords, color='red', marker='o')

# 각 노드에 번호 표시
for node_id, (x, y) in node_coordinates.items():
    plt.text(x, y, str(node_id), fontsize=8, ha='right')

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Node Coordinates Visualization')
plt.grid(True)
plt.show()
