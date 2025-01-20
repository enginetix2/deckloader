# Naval Air Vehicle Deck Optimization Tool

A Streamlit application that uses a genetic algorithm to optimize the placement and movement of vertical lift aircraft (e.g., V-22 Osprey) on a naval deck. The tool considers aircraft dimensions, deck size, overhang allowance, rotation, and movement order to find the most efficient configuration.

## Features

- **Aircraft Placement Optimization**: Determines optimal positions and orientations for aircraft on the deck.
- **Movement Order Optimization**: Identifies the best sequence for aircraft to move to the take-off area.
- **Aircraft Rotation**: Allows aircraft to be rotated to improve stacking efficiency on the deck.
- **Overhang Allowance**: Permits aircraft to extend over the edge of the deck by a specified allowance.
- **Visualization**: Displays the optimized deck layout with aircraft positions, orientations, movement paths, and movement order numbers.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/deck-optimization-tool.git
   cd deck-optimization-tool
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
   ```

3. **Install the Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

   If a `requirements.txt` file is not provided, install the packages manually:

   ```bash
   pip install streamlit deap matplotlib numpy shapely
   ```

## Usage

1. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

2. **Adjust Parameters**

   - Use the sidebar to set:
     - **Deck Length (m)**
     - **Deck Width (m)**
     - **Aircraft Length (m)**
     - **Aircraft Width (m)**
     - **Number of Aircraft**
     - **Overhang Allowance (m)**

3. **Run Optimization**

   - Click the **"Run Optimization"** button to start the genetic algorithm.
   - The app will display:
     - The best configuration fitness (penalty score).
     - A visualization of the optimized aircraft layout.

## How It Works

- **Genetic Algorithm**: The tool uses a genetic algorithm to minimize penalties associated with:
  - Overlapping parked aircraft.
  - Aircraft placed outside the deck bounds (considering overhang allowance).
  - Path collisions when moving aircraft to the take-off area.
- **Individual Representation**:
  - **Positions and Orientations**: Each aircraft's `x`, `y` coordinates and rotation angle `theta`.
  - **Moving Order**: A permutation of aircraft indices representing the sequence of movement.
- **Fitness Function**: Calculates a penalty score based on overlaps, boundary violations, and path collisions. The algorithm seeks to minimize this penalty.
- **Visualization**:
  - Aircraft are displayed as rotated rectangles.
  - Movement paths are shown as dashed green lines.
  - Movement order numbers are annotated on each aircraft.
  - The deck and take-off area are clearly marked.

## Requirements

- Python 3.6 or higher
- Packages:
  - Streamlit
  - DEAP
  - Matplotlib
  - NumPy
  - Shapely

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any improvements or bugs.

## Contact

For questions or feedback, please contact [enginetix@gmail.com].
