import random
import math
import os
import numpy as np
import matplotlib.pyplot as plt

ROLL_NUMBER = 63
SEED = ROLL_NUMBER

random.seed(SEED)
np.random.seed(SEED)

GRID_SIZE = random.randint(20, 25)
OBSTACLE_RATIO = 0.15
NUM_OBSTACLES = int(GRID_SIZE * GRID_SIZE * OBSTACLE_RATIO)

obstacles = set()

while len(obstacles) < NUM_OBSTACLES:
    obstacles.add((
        random.randint(0, GRID_SIZE - 1),
        random.randint(0, GRID_SIZE - 1)
    ))

free_cells = [
    (x, y)
    for x in range(GRID_SIZE)
    for y in range(GRID_SIZE)
    if (x, y) not in obstacles
]

start = random.choice(free_cells)

while True:
    goal = random.choice(free_cells)

    distance = math.sqrt(
        (goal[0] - start[0]) ** 2 +
        (goal[1] - start[1]) ** 2
    )

    if goal != start and distance >= GRID_SIZE * 0.5:
        break

NUM_PARTICLES = 60
NUM_WAYPOINTS = 5
ITERATIONS = 400

w = 0.6
c1 = 1.7
c2 = 1.7

particles = np.random.uniform(
    -GRID_SIZE / 3,
    GRID_SIZE / 3,
    (NUM_PARTICLES, NUM_WAYPOINTS, 2)
)

velocities = np.zeros_like(particles)

pbest_positions = particles.copy()
pbest_scores = np.full(NUM_PARTICLES, float("inf"))

gbest_position = None
gbest_score = float("inf")


def line_cells(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    cells = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    err = dx - dy

    while True:
        cells.append((x1, y1))

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x1 += sx

        if e2 < dx:
            err += dx
            y1 += sy

    return cells


def make_waypoints(particle):
    points = [start]

    for i in range(NUM_WAYPOINTS):
        t = (i + 1) / (NUM_WAYPOINTS + 1)

        base_x = start[0] + t * (goal[0] - start[0])
        base_y = start[1] + t * (goal[1] - start[1])

        x = base_x + particle[i][0]
        y = base_y + particle[i][1]

        x = int(round(np.clip(x, 0, GRID_SIZE - 1)))
        y = int(round(np.clip(y, 0, GRID_SIZE - 1)))

        points.append((x, y))

    points.append(goal)

    return points


def fitness(particle):
    path = make_waypoints(particle)

    total_distance = 0
    collisions = 0

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        total_distance += math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

        cells = line_cells(path[i], path[i + 1])

        for cell in cells:
            if cell in obstacles:
                collisions += 1

    return total_distance + collisions * 1000


for iteration in range(ITERATIONS):

    for i in range(NUM_PARTICLES):
        score = fitness(particles[i])

        if score < pbest_scores[i]:
            pbest_scores[i] = score
            pbest_positions[i] = particles[i].copy()

        if score < gbest_score:
            gbest_score = score
            gbest_position = particles[i].copy()

    for i in range(NUM_PARTICLES):
        r1 = np.random.random((NUM_WAYPOINTS, 2))
        r2 = np.random.random((NUM_WAYPOINTS, 2))

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest_positions[i] - particles[i])
            + c2 * r2 * (gbest_position - particles[i])
        )

        particles[i] += velocities[i]

        particles[i] = np.clip(
            particles[i],
            -GRID_SIZE / 2,
            GRID_SIZE / 2
        )

    if iteration % 20 == 0:
        print(
            "Iteration:",
            iteration,
            "Best Cost:",
            round(gbest_score, 2)
        )

best_waypoints = make_waypoints(gbest_position)

final_path = []

for i in range(len(best_waypoints) - 1):
    segment = line_cells(
        best_waypoints[i],
        best_waypoints[i + 1]
    )

    if i > 0:
        segment = segment[1:]

    final_path.extend(segment)

path_length = 0

for i in range(len(final_path) - 1):
    x1, y1 = final_path[i]
    x2, y2 = final_path[i + 1]

    path_length += math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )

print("\nSeed:", SEED)
print("Grid Size:", GRID_SIZE, "x", GRID_SIZE)
print("Start:", start)
print("Goal:", goal)
print("Obstacles:", len(obstacles))
print("Best Waypoints:", best_waypoints)
print("Best Cost:", round(gbest_score, 2))
print("Path Length:", round(path_length, 2))

collision_found = any(
    cell in obstacles
    for cell in final_path
)

print("Obstacle Free:", not collision_found)

plt.figure(figsize=(9, 9))

for x, y in obstacles:
    plt.scatter(
        x,
        y,
        marker="s",
        s=90,
        color="black"
    )

plt.scatter(
    start[0],
    start[1],
    s=180,
    color="green",
    label="Start",
    zorder=5
)

plt.scatter(
    goal[0],
    goal[1],
    s=220,
    marker="*",
    color="red",
    label="Goal",
    zorder=5
)

path_x = [p[0] for p in final_path]
path_y = [p[1] for p in final_path]

plt.plot(
    path_x,
    path_y,
    color="blue",
    linewidth=2.5,
    label="Best Path",
    zorder=4
)

plt.xlim(-0.5, GRID_SIZE - 0.5)
plt.ylim(-0.5, GRID_SIZE - 0.5)

plt.xticks(range(GRID_SIZE))
plt.yticks(range(GRID_SIZE))

plt.grid(True)

plt.title(
    f"PSO Path Planning - Seed {SEED}\n"
    f"Cost = {gbest_score:.2f}"
)

plt.legend()
plt.tight_layout()

os.makedirs("images", exist_ok=True)

plt.savefig(
    "images/final_path.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()