import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# Streamlit App Setup
st.title("Naval Air Vehicle Deck Optimization Tool")

# Input Parameters
deck_length = st.sidebar.number_input("Deck Length (m)", value=100)
deck_width = st.sidebar.number_input("Deck Width (m)", value=20)
aircraft_length = st.sidebar.number_input("Aircraft Length (m)", value=15)
aircraft_width = st.sidebar.number_input("Aircraft Width (m)", value=5)
# Ensure num_aircraft is at least 2 to prevent crossover issues
num_aircraft = st.sidebar.number_input("Number of Aircraft", value=10, min_value=2)

# Genetic Algorithm Setup
POP_SIZE = 50
NGEN = 100
MUTPB = 0.2
CXPB = 0.5

# Define Individual and Fitness Function
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual():
    # Each individual represents the x, y coordinates of each aircraft
    individual = []
    for _ in range(num_aircraft):
        x = random.uniform(0, deck_length - aircraft_length)
        y = random.uniform(0, deck_width - aircraft_width)
        individual.extend([x, y])
    return individual

def evaluate(individual):
    total_aircraft = len(individual) // 2
    penalty = 0

    for i in range(total_aircraft):
        x1, y1 = individual[i*2], individual[i*2+1]
        for j in range(i + 1, total_aircraft):
            x2, y2 = individual[j*2], individual[j*2+1]
            # Check for overlap between aircraft
            if abs(x1 - x2) < aircraft_length and abs(y1 - y2) < aircraft_width:
                penalty += 1

    return total_aircraft - penalty,

def custom_mutation(individual, mu=0.0, sigma=5.0, indpb=0.2):
    """Applies Gaussian mutation to each element in the individual with probability indpb."""
    for i in range(len(individual)):
        if random.random() < indpb:
            # Apply Gaussian mutation and ensure the individual stays within bounds
            if i % 2 == 0:  # Mutating x-coordinate
                individual[i] = min(max(individual[i] + random.gauss(mu, sigma), 0), deck_length - aircraft_length)
            else:  # Mutating y-coordinate
                individual[i] = min(max(individual[i] + random.gauss(mu, sigma), 0), deck_width - aircraft_width)
    return individual,

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", custom_mutation, mu=0.0, sigma=5.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def run_ga():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, 
                        stats=None, halloffame=hof, verbose=False)

    return hof[0]

def visualize_layout(individual):
    fig, ax = plt.subplots()
    ax.set_xlim(0, deck_length)
    ax.set_ylim(0, deck_width)
    ax.set_aspect('equal')

    total_aircraft = len(individual) // 2
    for i in range(total_aircraft):
        x, y = individual[i*2], individual[i*2+1]
        rect = plt.Rectangle((x, y), aircraft_length, aircraft_width, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(rect)
    
    ax.set_title("Optimized Aircraft Layout")
    st.pyplot(fig)

if st.button("Run Optimization"):
    st.write("Running Genetic Algorithm...")
    best_individual = run_ga()
    st.write(f"Best Configuration: {best_individual}")
    visualize_layout(best_individual)
