import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from shapely.geometry import LineString, Polygon, box
from shapely.affinity import rotate, translate

# Streamlit App Setup
st.title("Naval Air Vehicle Deck Optimization Tool with Rotation and Moving Order")

# Input Parameters
deck_length = st.sidebar.number_input("Deck Length (m)", value=100)
deck_width = st.sidebar.number_input("Deck Width (m)", value=20)
aircraft_length = st.sidebar.number_input("Aircraft Length (m)", value=15)
aircraft_width = st.sidebar.number_input("Aircraft Width (m)", value=5)
# Ensure num_aircraft is at least 2 to prevent crossover issues
num_aircraft = st.sidebar.number_input("Number of Aircraft", value=10, min_value=2)
overhang_allowance = st.sidebar.number_input("Overhang Allowance (m)", value=2.0, min_value=0.0)

# Genetic Algorithm Setup
POP_SIZE = 50
NGEN = 100
MUTPB = 0.2
CXPB = 0.5

# Define Take-off Area (e.g., a rectangular area at one end of the deck)
takeoff_area_x = deck_length - aircraft_length  # Position at the far end of the deck
takeoff_area_y = deck_width / 2 - aircraft_width / 2  # Centered vertically

# Define Individual and Fitness Function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize penalty
creator.create("Individual", list, fitness=creator.FitnessMin)

positions_length = 3 * num_aircraft  # x, y, theta for each aircraft

def create_individual():
    individual = []
    # Positions and orientations
    for _ in range(num_aircraft):
        x = random.uniform(-overhang_allowance, deck_length - aircraft_length + overhang_allowance)
        y = random.uniform(-overhang_allowance, deck_width - aircraft_width + overhang_allowance)
        theta = random.uniform(0, 360)
        individual.extend([x, y, theta])
    # Moving order as a permutation of aircraft indices
    moving_order = list(range(num_aircraft))
    random.shuffle(moving_order)
    individual.extend(moving_order)
    return individual

def evaluate(individual):
    positions_orientations = individual[:positions_length]
    moving_order = [int(i) for i in individual[positions_length:]]

    total_aircraft = num_aircraft
    penalty = 0

    # Create deck polygon with overhang allowance
    deck_with_overhang = box(-overhang_allowance, -overhang_allowance,
                             deck_length + overhang_allowance, deck_width + overhang_allowance)

    # Create aircraft polygons
    aircraft_polygons = []
    for i in range(total_aircraft):
        idx = i * 3
        x, y, theta = positions_orientations[idx], positions_orientations[idx + 1], positions_orientations[idx + 2]
        # Create aircraft polygon
        rect = box(0, 0, aircraft_length, aircraft_width)
        rotated_rect = rotate(rect, theta, origin=(aircraft_length / 2, aircraft_width / 2), use_radians=False)
        # Translate to position
        aircraft_poly = translate(rotated_rect, x, y)
        aircraft_polygons.append(aircraft_poly)

        # Check if aircraft is within acceptable area (including overhang allowance)
        if not deck_with_overhang.contains(aircraft_poly):
            penalty += 1  # Penalize if aircraft is outside acceptable area

    # Check for overlapping parked aircraft
    for i in range(total_aircraft):
        for j in range(i + 1, total_aircraft):
            if aircraft_polygons[i].intersects(aircraft_polygons[j]):
                penalty += 1

    # Simulate aircraft movement in moving order
    moved_aircraft_polygons = []
    for idx in moving_order:
        # Path from aircraft to take-off area
        idx3 = idx * 3
        x, y, theta = positions_orientations[idx3], positions_orientations[idx3 + 1], positions_orientations[idx3 + 2]
        aircraft_poly = aircraft_polygons[idx]

        path = LineString([(x + aircraft_length / 2, y + aircraft_width / 2),
                           (takeoff_area_x + aircraft_length / 2, takeoff_area_y + aircraft_width / 2)])
        # Check for collisions with parked aircraft (excluding self and moved aircraft)
        for j in range(total_aircraft):
            if j != idx and aircraft_polygons[j] not in moved_aircraft_polygons:
                if path.intersects(aircraft_polygons[j]):
                    penalty += 1
                    break  # Only count one penalty per aircraft

        # Move aircraft to take-off area (remove from parked aircraft)
        moved_aircraft_polygons.append(aircraft_polygons[idx])

    return (penalty,)

def custom_mutation(individual, mu=0.0, sigma=5.0, indpb=0.2):
    # Mutate positions and orientations
    for i in range(positions_length):
        if random.random() < indpb:
            if i % 3 == 0:  # x-coordinate
                individual[i] = min(max(individual[i] + random.gauss(mu, sigma), -overhang_allowance),
                                    deck_length - aircraft_length + overhang_allowance)
            elif i % 3 == 1:  # y-coordinate
                individual[i] = min(max(individual[i] + random.gauss(mu, sigma), -overhang_allowance),
                                    deck_width - aircraft_width + overhang_allowance)
            else:  # theta
                individual[i] = (individual[i] + random.gauss(mu, sigma)) % 360
    # Mutate moving order
    perm = individual[positions_length:]
    if random.random() < indpb:
        tools.mutShuffleIndexes(perm, indpb=1.0)
    individual[positions_length:] = perm
    return individual,

def custom_crossover(ind1, ind2):
    # Crossover positions and orientations
    for i in range(positions_length):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    # Crossover moving order
    perm1 = ind1[positions_length:]
    perm2 = ind2[positions_length:]
    tools.cxOrdered(perm1, perm2)
    ind1[positions_length:] = perm1
    ind2[positions_length:] = perm2
    return ind1, ind2

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", custom_crossover)
toolbox.register("mutate", custom_mutation)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def run_ga():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, 
                        stats=None, halloffame=hof, verbose=False)

    return hof[0]

def visualize_layout(individual):
    positions_orientations = individual[:positions_length]
    moving_order = [int(i) for i in individual[positions_length:]]

    fig, ax = plt.subplots()
    ax.set_xlim(-overhang_allowance, deck_length + overhang_allowance)
    ax.set_ylim(-overhang_allowance, deck_width + overhang_allowance)
    ax.set_aspect('equal')

    # Draw the deck
    deck = plt.Rectangle((0, 0), deck_length, deck_width, edgecolor='black', facecolor='gray', alpha=0.3)
    ax.add_patch(deck)

    # Draw the take-off area
    takeoff_area = plt.Rectangle((takeoff_area_x, takeoff_area_y), aircraft_length, aircraft_width,
                                 edgecolor='red', facecolor='salmon', alpha=0.5)
    ax.add_patch(takeoff_area)
    ax.text(takeoff_area_x + aircraft_length / 2, takeoff_area_y + aircraft_width / 2,
            'Take-off Area', color='white', ha='center', va='center')

    total_aircraft = num_aircraft
    aircraft_polygons = []

    # Draw parked aircraft
    for i in range(total_aircraft):
        idx = i * 3
        x, y, theta = positions_orientations[idx], positions_orientations[idx + 1], positions_orientations[idx + 2]
        # Create aircraft rectangle
        rect = plt.Rectangle((0, 0), aircraft_length, aircraft_width, edgecolor='blue', facecolor='lightblue')
        t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(aircraft_length / 2, aircraft_width / 2, theta) + \
            plt.matplotlib.transforms.Affine2D().translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        # Add movement order number
        ax.text(x + aircraft_length / 2, y + aircraft_width / 2, f"{moving_order.index(i)+1}",
                ha='center', va='center', fontsize=8, color='black')

        # Store aircraft polygon for path calculations
        poly = box(0, 0, aircraft_length, aircraft_width)
        rotated_poly = rotate(poly, theta, origin=(aircraft_length / 2, aircraft_width / 2), use_radians=False)
        aircraft_poly = translate(rotated_poly, x, y)
        aircraft_polygons.append(aircraft_poly)

    # Draw movement paths in moving order
    for idx in moving_order:
        idx3 = idx * 3
        x, y, theta = positions_orientations[idx3], positions_orientations[idx3 + 1], positions_orientations[idx3 + 2]
        path = LineString([(x + aircraft_length / 2, y + aircraft_width / 2),
                           (takeoff_area_x + aircraft_length / 2, takeoff_area_y + aircraft_width / 2)])
        x_path, y_path = path.xy
        ax.plot(x_path, y_path, color='green', linestyle='--')

    ax.set_title("Optimized Aircraft Layout with Rotation and Moving Order")
    st.pyplot(fig)

if st.button("Run Optimization"):
    st.write("Running Genetic Algorithm...")
    best_individual = run_ga()
    st.write(f"Best Configuration Fitness (Penalty): {best_individual.fitness.values[0]}")
    visualize_layout(best_individual)
