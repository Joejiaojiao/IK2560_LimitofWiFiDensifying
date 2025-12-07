import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
AREA_SIZE = 1000  # Size of the area (meters)
NUM_USERS = 100  # Number of users
MIN_DISTANCE = 2  # Minimum distance between APs (meters)
AP_POWER = 20  # Transmit power of APs (dBm)
NOISE_POWER = -90  # Noise power (dBm)
BANDWIDTH = 20  # Bandwidth (MHz)
SPECTRAL_EFFICIENCY = 2  # Bits per Hz of bandwidth
NUM_SIMULATIONS = 100  # Number of Monte Carlo simulations
MAX_COVERAGE_RADIUS = AREA_SIZE / 5  # Maximum coverage radius (20% of the area)
MIN_COVERAGE_RADIUS = AREA_SIZE / 20  # Minimum coverage radius (5% of the area)

# ---------------------- AP Placement Function ----------------------
def grid_generate_aps(area_size, grid_size, min_radius, max_radius):
    """
    Generate APs in a grid pattern with random coverage radii.
    :param area_size: Size of the square area (meters).
    :param grid_size: Number of APs along one dimension (grid resolution).
    :param min_radius: Minimum coverage radius for APs.
    :param max_radius: Maximum coverage radius for APs.
    :return: Array of AP positions (x, y) and coverage radii.
    """
    x_coords = np.linspace(0, area_size, grid_size)
    y_coords = np.linspace(0, area_size, grid_size)
    ap_positions = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    coverage_radii = np.random.uniform(min_radius, max_radius, size=len(ap_positions))
    return ap_positions, coverage_radii

# ---------------------- User Position Generation ----------------------
def generate_user_positions(num_users, area_size):
    """
    Generate random user positions uniformly distributed in the area.
    :param num_users: Number of users.
    :param area_size: Size of the square area (meters).
    :return: Array of user positions (x, y).
    """
    return np.random.uniform(0, area_size, size=(num_users, 2))

# ---------------------- User-to-AP Assignment ----------------------
def assign_users_to_aps_signal(user_positions, ap_positions, ap_power, alpha=3, shadowing_sigma=8):
    """
    Assign users to the AP with the strongest signal strength.
    :param user_positions: Positions of the users.
    :param ap_positions: Positions of the APs.
    :param ap_power: Transmit power of the APs (dBm).
    :param alpha: Path loss exponent.
    :param shadowing_sigma: Standard deviation of shadowing (dB).
    :return: Assigned AP indices and signal strengths.
    """
    num_users = len(user_positions)
    num_aps = len(ap_positions)
    signal_strengths = np.zeros((num_users, num_aps))

    for i, user_pos in enumerate(user_positions):
        for j, ap_pos in enumerate(ap_positions):
            distance = np.linalg.norm(user_pos - ap_pos)
            path_loss = 10 * alpha * np.log10(distance + 1e-3)  # Path loss in dB
            shadowing = np.random.normal(0, shadowing_sigma)  # Log-normal shadowing
            signal_strengths[i, j] = ap_power - path_loss + shadowing  # Signal strength in dBm

    assigned_aps = np.argmax(signal_strengths, axis=1)  # Assign to AP with strongest signal
    return assigned_aps, signal_strengths

# ---------------------- SINR and Throughput Calculation ----------------------
def calculate_sinr_and_throughput(user_positions, ap_positions, assigned_aps, ap_power, noise_power, alpha=3, shadowing_sigma=8):

    num_users = len(user_positions)
    signal_power = np.zeros(num_users)
    interference_power = np.zeros(num_users)

    for i, user_pos in enumerate(user_positions):
        serving_ap = ap_positions[assigned_aps[i]]
        distance = np.linalg.norm(user_pos - serving_ap)
        path_loss = 10 * alpha * np.log10(distance + 1e-3)
        shadowing = np.random.normal(0, shadowing_sigma)
        fading = np.random.rayleigh(scale=1)  # Rayleigh fading

        signal_power[i] = ap_power - path_loss + shadowing + 10 * np.log10(fading)

        interfering_aps = np.delete(ap_positions, assigned_aps[i], axis=0)
        interference = []
        for interfering_ap in interfering_aps:
            distance_interf = np.linalg.norm(user_pos - interfering_ap)
            path_loss_interf = 10 * alpha * np.log10(distance_interf + 1e-3)
            shadowing_interf = np.random.normal(0, shadowing_sigma)
            fading_interf = np.random.rayleigh(scale=1)
            interference_power_dBm = ap_power - path_loss_interf + shadowing_interf + 10 * np.log10(fading_interf)
            interference.append(10 ** (interference_power_dBm / 10))  # Convert to linear scale

        interference_power[i] = np.sum(interference)

    noise_power_linear = 10 ** (noise_power / 10)
    signal_power_linear = 10 ** (signal_power / 10)
    sinr_linear = signal_power_linear / (noise_power_linear + interference_power)
    sinr_db = 10 * np.log10(sinr_linear)
    throughput = BANDWIDTH * SPECTRAL_EFFICIENCY * np.log2(1 + sinr_linear)  # Shannon's formula

    return np.mean(throughput), np.mean(sinr_db)

# ---------------------- Plot AP Coverage ----------------------
def plot_ap_coverage(ap_positions, coverage_radii, area_size):
    """
    Visualize AP positions and their coverage areas with outlined circles.
    :param ap_positions: Array of AP positions (x, y).
    :param coverage_radii: Array of coverage radii for APs.
    :param area_size: Size of the simulation area (meters).
    """
    plt.figure(figsize=(8, 8))
    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.gca().set_aspect('equal', adjustable='box')

    for ap, radius in zip(ap_positions, coverage_radii):
        circle = plt.Circle(ap, radius, color='blue', fill=False, linestyle='--', linewidth=1.5)
        plt.gca().add_patch(circle)
        plt.plot(ap[0], ap[1], 'bo', label='Access Point')

    plt.title("Access Points and Coverage Areas")
    plt.xlabel("X-coordinate (meters)")
    plt.ylabel("Y-coordinate (meters)")
    plt.grid(True)
    plt.show()

# ---------------------- Monte Carlo Simulation ----------------------
def monte_carlo_simulation(ap_density):
    """
    Perform Monte Carlo simulation for a given AP density.
    :param ap_density: AP density (APs per square meter).
    :return: Average throughput and SINR for the given density.
    """
    grid_size = int(np.sqrt(ap_density * AREA_SIZE ** 2))
    ap_positions, coverage_radii = grid_generate_aps(AREA_SIZE, grid_size, MIN_COVERAGE_RADIUS, MAX_COVERAGE_RADIUS)
    throughput_results = []
    sinr_results = []

    for _ in range(NUM_SIMULATIONS):
        user_positions = generate_user_positions(NUM_USERS, AREA_SIZE)
        assigned_aps, _ = assign_users_to_aps_signal(user_positions, ap_positions, AP_POWER)
        avg_throughput, avg_sinr = calculate_sinr_and_throughput(user_positions, ap_positions, assigned_aps, AP_POWER, NOISE_POWER)
        throughput_results.append(avg_throughput)
        sinr_results.append(avg_sinr)

    return np.mean(throughput_results), np.mean(sinr_results), ap_positions, coverage_radii

# ---------------------- Main Function ----------------------
def main():
    """
    Main function to execute the simulation and visualize results.
    """
    ap_densities = np.linspace(0.00001, 0.001, 50)  # AP densities (AP/m²)
    throughput_results = []
    sinr_results = []

    for density in ap_densities:
        print(f"Simulating for AP Density: {density:.3f} AP/m²...")
        avg_throughput, avg_sinr, _, _ = monte_carlo_simulation(density)
        throughput_results.append(avg_throughput)
        sinr_results.append(avg_sinr)

    # Visualization of results
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("AP Density (AP/m²)")
    ax1.set_ylabel("Average Throughput (Mbps)", color='tab:blue')
    ax1.plot(ap_densities, throughput_results, marker='o', linestyle='-', color='tab:blue', label='Throughput')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average SINR (dB)", color='tab:red')
    ax2.plot(ap_densities, sinr_results, marker='x', linestyle='--', color='tab:red', label='SINR')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title("Impact of AP Density on Throughput and SINR")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
