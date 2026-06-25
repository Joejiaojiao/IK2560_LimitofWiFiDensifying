# Limit of Densifying WiFi

This project simulates how WiFi network performance changes when the density of Access Points (APs) increases. The main goal is to observe the relationship between **AP density**, **SINR**, and **average user throughput**.

The code is simulation-based. It does not scan real WiFi signals or use real network devices.

## Project Overview

The project builds a simplified WiFi deployment scenario:

1. Generate APs in a fixed area.
2. Generate users randomly in the same area.
3. Assign each user to an AP.
4. Calculate signal strength, interference, SINR, and throughput.
5. Repeat the experiment under different AP densities.
6. Plot how throughput and SINR change as the network becomes denser.

Through this process, the project studies whether adding more APs always improves performance.

## Main Result

The simulation shows that increasing AP density improves throughput at the beginning, because users are closer to APs and receive stronger signals.

However, when AP density becomes too high, additional APs create more interference. As a result, the throughput stops increasing significantly and reaches a saturation stage. This means that WiFi densification has a practical limit: more APs are useful only before interference becomes the dominant factor.

## Files

```text
.
├── MCSimulation.py
├── MassivAP.py
├── channelsimmulation.py
└── addon.py
```

### `MCSimulation.py`

Main Monte Carlo simulation script. It places APs in a grid, generates random users, assigns users to the AP with the strongest received signal, calculates SINR and throughput, and plots the final result.

### `MassivAP.py`

A simplified simulation using an SSI-style AP generation method. It generates APs with a minimum distance constraint, assigns users to the nearest AP, and evaluates throughput and SINR.

### `channelsimmulation.py`

A fixed-frequency channel simulation. It uses free-space path loss to calculate received signal power, interference, SINR, and throughput.

### `addon.py`

An exploratory version that links AP coverage radius with frequency. This file represents an earlier modeling attempt and is kept for reference.

## Requirements

Install the required Python packages:

```bash
pip install numpy matplotlib
```

Recommended Python version:

```text
Python 3.8+
```

## How to Run

Run the main simulation:

```bash
python MCSimulation.py
```

You can also run the other scripts separately:

```bash
python MassivAP.py
python channelsimmulation.py
python addon.py
```

Each script will print the current AP density being simulated and then display a plot showing the relationship between AP density, throughput, and SINR.

## Output

The main output is a Matplotlib figure with:

* AP density on the x-axis
* average throughput on one y-axis
* average SINR on the other y-axis

The figure is used to show the performance trend under different WiFi deployment densities.

## Notes

This project is a simplified research simulation. Some real-world WiFi factors are not fully modeled, such as wall loss, indoor layout, MAC-layer contention, dynamic channel allocation, and realistic user mobility.

The current code is mainly used to demonstrate the basic trend of dense WiFi deployment: AP densification improves performance at first, but excessive density leads to stronger interference and throughput saturation.
