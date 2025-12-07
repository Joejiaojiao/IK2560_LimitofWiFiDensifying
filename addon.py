import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
AREA_SIZE = 5000  # Siz
# e of the area (meters)
NUM_USERS = 200  # Number of users
AP_POWER = 100  # Transmit power of APs (mW)
NOISE_POWER = 1e-9  # Noise power (W)
BANDWIDTH = 20e6  # Bandwidth (Hz)
NUM_SIMULATIONS = 100  # Number of Monte Carlo simulations
MAX_COVERAGE_RADIUS = 300  # Maximum coverage radius (meters)
MIN_COVERAGE_RADIUS = 150  # Minimum coverage radius (meters)
C = 3e8  # Speed of light (m/s)
DEFAULT_FREQUENCY = 2.4e9  # Default reference frequency (Hz)

# ---------------------- Path Loss Based Frequency Calculation ----------------------
def calculate_ap_frequency(coverage_radius, transmit_power_dbm, min_rx_power_dbm, path_loss_exponent=3):
    """
    根据路径损耗模型计算 AP 适合的频率
    """
    d0 = 1

    # 计算路径损耗中的参考损耗
    20 * np.log10(4 * np.pi * d0 * DEFAULT_FREQUENCY / C)
    additional_loss = 10 * path_loss_exponent * np.log10(coverage_radius / d0)

    # 计算适当的频率
    exponent = (transmit_power_dbm - min_rx_power_dbm - additional_loss) / 20
    frequency = (C / (4 * np.pi * coverage_radius)) * (10 ** exponent)
    return frequency

# ---------------------- AP Placement ----------------------
def grid_generate_aps(area_size, grid_size, min_radius, max_radius, transmit_power_dbm, min_rx_power_dbm):
    """
    生成 AP 位置，并根据路径损耗计算合适的频率
    """
    x_coords = np.linspace(0, area_size, grid_size)
    y_coords = np.linspace(0, area_size, grid_size)
    ap_positions = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    coverage_radii = np.random.uniform(min_radius, max_radius, size=len(ap_positions))
    ap_frequencies = np.array([calculate_ap_frequency(R, transmit_power_dbm, min_rx_power_dbm) for R in coverage_radii])

    return ap_positions, coverage_radii, ap_frequencies

# ---------------------- User Position Generation ----------------------
def generate_user_positions(num_users, area_size):
    """
    Generate random user positions uniformly distributed in the area.
    """
    return np.random.uniform(0, area_size, size=(num_users, 2))

# ---------------------- Path Loss Calculation ----------------------
def free_space_path_loss(distance, frequency):
    """
    Calculate free-space path loss based on distance and frequency.
    """
    wavelength = C / frequency
    return (wavelength / (4 * np.pi * distance)) ** 2

# ---------------------- User-to-AP Assignment ----------------------
def assign_users_to_aps_signal(user_positions, ap_positions, ap_frequencies, transmit_power):
    """
    Assign users to the AP with the strongest signal strength.
    """
    num_users = len(user_positions)
    num_aps = len(ap_positions)
    signal_strengths = np.zeros((num_users, num_aps))

    for i, user_pos in enumerate(user_positions):
        for j, ap_pos in enumerate(ap_positions):
            distance = np.linalg.norm(user_pos - ap_pos)
            if distance < 1e-3:
                distance = 1e-3  # Prevent division by zero
            path_loss = free_space_path_loss(distance, ap_frequencies[j])
            signal_strengths[i, j] = transmit_power * path_loss

    assigned_aps = np.argmax(signal_strengths, axis=1)
    return assigned_aps, signal_strengths

# ---------------------- SINR and Throughput Calculation ----------------------
def calculate_sinr_and_throughput(user_positions, ap_positions, assigned_aps, ap_frequencies, coverage_radii, transmit_power,
                                  noise_power, bandwidth):
    """
    Calculate SINR and throughput for each user.
    """
    num_users = len(user_positions)
    sinrs = []
    throughputs = []

    for i, user_pos in enumerate(user_positions):
        serving_ap = ap_positions[assigned_aps[i]]
        serving_frequency = ap_frequencies[assigned_aps[i]]
        distance = np.linalg.norm(user_pos - serving_ap)
        if distance < 1e-3:
            distance = 1e-3
        path_loss = free_space_path_loss(distance, serving_frequency)
        signal_power = transmit_power * path_loss

        interference_power = 0
        for j, ap_pos in enumerate(ap_positions):
            if j != assigned_aps[i]:
                interfer_distance = np.linalg.norm(user_pos - ap_pos)
                if interfer_distance < 1e-3:
                    interfer_distance = 1e-3

                # Check if the AP covers the user
                if interfer_distance <= coverage_radii[j]:
                    interfer_frequency = ap_frequencies[j]
                    interfer_path_loss = free_space_path_loss(interfer_distance, interfer_frequency)
                    interference_power += transmit_power * interfer_path_loss

        sinr = signal_power / (interference_power + noise_power)
        throughput = bandwidth * np.log2(1 + sinr)

        sinrs.append(sinr)
        throughputs.append(throughput)

    return np.mean(throughputs), np.mean(sinrs)

# ---------------------- Monte Carlo Simulation ----------------------
def monte_carlo_simulation(ap_density):
    """
    Perform Monte Carlo simulation for a given AP density.
    """
    grid_size = int(np.sqrt(ap_density * AREA_SIZE ** 2))
    AP_TX_POWER_DBM = 20
    MIN_RX_POWER_DBM = -90

    ap_positions, coverage_radii, ap_frequencies = grid_generate_aps(AREA_SIZE, grid_size, MIN_COVERAGE_RADIUS,
                                                                     MAX_COVERAGE_RADIUS, AP_TX_POWER_DBM,
                                                                     MIN_RX_POWER_DBM)
    throughput_results = []
    sinr_results = []

    for _ in range(NUM_SIMULATIONS):
        user_positions = generate_user_positions(NUM_USERS, AREA_SIZE)
        assigned_aps, _ = assign_users_to_aps_signal(user_positions, ap_positions, ap_frequencies, AP_POWER)
        avg_throughput, avg_sinr = calculate_sinr_and_throughput(user_positions, ap_positions, assigned_aps,
                                                                 ap_frequencies, coverage_radii, AP_POWER, NOISE_POWER, BANDWIDTH)
        throughput_results.append(avg_throughput)
        sinr_results.append(avg_sinr)

    return np.mean(throughput_results), np.mean(sinr_results)

# ---------------------- Main Function ----------------------
def main():
    """
    Main function to execute the simulation and visualize results.
    """
    ap_densities = np.linspace(0.00001, 0.000025, 20)  # AP densities (AP/m²)
    throughput_results = []
    sinr_results = []

    for density in ap_densities:
        print(f"Simulating for AP Density: {density:.5f} AP/m²...")
        avg_throughput, avg_sinr = monte_carlo_simulation(density)
        throughput_results.append(avg_throughput)
        sinr_results.append(avg_sinr)

    # Visualization of results
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("AP Density (AP/m²)")
    ax1.set_ylabel("Average Throughput (bps)", color='tab:blue')
    ax1.plot(ap_densities, throughput_results, marker='o', linestyle='-', color='tab:blue', label='Throughput')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average SINR", color='tab:red')
    ax2.plot(ap_densities, sinr_results, marker='x', linestyle='--', color='tab:red', label='SINR')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title("Impact of AP Density on Throughput and SINR")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
