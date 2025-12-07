import numpy as np
import matplotlib.pyplot as plt

# 仿真参数
AREA_SIZE = 100  # 区域大小 (m)
NUM_USERS = 100  # 用户数量
MIN_DISTANCE = 2  # AP 之间的最小距离 (m)
AP_POWER = 20  # AP 发射功率 (dBm)
NOISE_POWER = -90  # 噪声功率 (dBm)
BANDWIDTH = 20  # 带宽 (MHz)
SPECTRAL_EFFICIENCY = 2  # 每 Hz 的比特速率 (bps/Hz)
NUM_SIMULATIONS = 100  # Monte Carlo 仿真次数

# ---------------------- SSI AP 位置生成函数 ----------------------
def ssi_generate_aps(num_aps, area_size, min_distance):
    """
    使用 Simple Sequential Inhibition (SSI) 方法生成 AP 位置
    """
    ap_positions = []  # 存储 AP 位置
    attempts = 0  # 尝试次数
    max_attempts = num_aps * 100

    while len(ap_positions) < num_aps and attempts < max_attempts:
        candidate = np.random.uniform(0, area_size, size=(2,))
        attempts += 1
        # 检查与已有 AP 的最小距离
        if all(np.linalg.norm(candidate - np.array(ap)) >= min_distance for ap in ap_positions):
            ap_positions.append(candidate)

    if len(ap_positions) < num_aps:
        print(f"Warning: Only {len(ap_positions)} APs generated after {max_attempts} attempts.")

    return np.array(ap_positions)

# ---------------------- 用户位置生成 ----------------------
def generate_user_positions(num_users, area_size):
    """
    随机生成用户位置
    """
    return np.random.uniform(0, area_size, size=(num_users, 2))

# ---------------------- 用户分配到最近的 AP ----------------------
def assign_users_to_aps(user_positions, ap_positions):
    """
    分配用户到最近的 AP，并计算距离
    """
    distances = np.linalg.norm(user_positions[:, np.newaxis] - ap_positions, axis=2)
    assigned_aps = np.argmin(distances, axis=1)
    return assigned_aps, distances

# ---------------------- SINR 和吞吐量计算 ----------------------
def calculate_sinr_and_throughput(user_positions, ap_positions, assigned_aps, ap_power, noise_power, alpha=3, shadowing_sigma=8):
    """
    计算每个用户的 SINR 和吞吐量，考虑路径损耗、阴影衰落和快速衰落。
    :param user_positions: 用户位置
    :param ap_positions: AP 位置
    :param assigned_aps: 每个用户分配的最近 AP 索引
    :param ap_power: AP 发射功率 (dBm)
    :param noise_power: 噪声功率 (dBm)
    :param alpha: 路径损耗指数
    :param shadowing_sigma: 阴影衰落标准差 (dB)
    """
    # 初始化变量
    num_users = len(user_positions)
    signal_power = np.zeros(num_users)
    interference_power = np.zeros(num_users)

    # 遍历每个用户，计算信号功率和干扰功率
    for i, user_pos in enumerate(user_positions):
        # 距离最近的 AP
        serving_ap = ap_positions[assigned_aps[i]]
        distance = np.linalg.norm(user_pos - serving_ap)

        # 添加路径损耗和阴影衰落
        path_loss = 10 * alpha * np.log10(distance + 1e-3)
        shadowing = np.random.normal(0, shadowing_sigma)
        fading = np.random.rayleigh(scale=1)  # 快速衰落（Rayleigh 分布）

        # 计算信号功率 (dBm)
        signal_power[i] = ap_power - path_loss + shadowing + 10 * np.log10(fading)

        # 计算来自其他 AP 的干扰功率
        interfering_aps = np.delete(ap_positions, assigned_aps[i], axis=0)
        interference = []
        for interfering_ap in interfering_aps:
            distance_interf = np.linalg.norm(user_pos - interfering_ap)
            path_loss_interf = 10 * alpha * np.log10(distance_interf + 1e-3)
            shadowing_interf = np.random.normal(0, shadowing_sigma)
            fading_interf = np.random.rayleigh(scale=1)
            interference_power_dBm = ap_power - path_loss_interf + shadowing_interf + 10 * np.log10(fading_interf)
            interference.append(10 ** (interference_power_dBm / 10))  # 转换为线性功率

        interference_power[i] = np.sum(interference)

    # 噪声功率（线性）
    noise_power_linear = 10 ** (noise_power / 10)

    # 计算 SINR 和吞吐量
    signal_power_linear = 10 ** (signal_power / 10)
    sinr_linear = signal_power_linear / (noise_power_linear + interference_power)
    sinr_db = 10 * np.log10(sinr_linear)

    # 通过 Shannon 容量公式计算吞吐量
    throughput = BANDWIDTH * SPECTRAL_EFFICIENCY * np.log2(1 + sinr_linear)

    return np.mean(throughput), np.mean(sinr_db)

# ---------------------- Monte Carlo 仿真 ----------------------
def monte_carlo_simulation(ap_density):
    """
    进行 Monte Carlo 仿真，基于不同的 AP 密度计算平均吞吐量和 SINR
    """
    num_aps = int(ap_density * (AREA_SIZE ** 2))
    throughput_results = []
    sinr_results = []

    for _ in range(NUM_SIMULATIONS):
        ap_positions = ssi_generate_aps(num_aps, AREA_SIZE, MIN_DISTANCE)
        user_positions = generate_user_positions(NUM_USERS, AREA_SIZE)
        assigned_aps, _ = assign_users_to_aps(user_positions, ap_positions)
        avg_throughput, avg_sinr = calculate_sinr_and_throughput(user_positions, ap_positions, assigned_aps, AP_POWER, NOISE_POWER)
        throughput_results.append(avg_throughput)
        sinr_results.append(avg_sinr)

    return np.mean(throughput_results), np.mean(sinr_results)

# ---------------------- 仿真主函数 ----------------------
def main():
    """
    主函数：执行仿真并绘制结果
    """
    ap_densities = np.linspace(0.01, 0.1, 10)  # AP 密度范围 (AP/m²)
    throughput_results = []
    sinr_results = []

    # 依次仿真不同 AP 密度
    for density in ap_densities:
        print(f"Simulating for AP Density: {density:.3f} AP/m²...")
        avg_throughput, avg_sinr = monte_carlo_simulation(density)
        throughput_results.append(avg_throughput)
        sinr_results.append(avg_sinr)

    # 绘制结果
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