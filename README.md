# V2X Traffic Safety System Simulation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-green.svg)](https://matplotlib.org/)

## Overview

A Python-based simulation that compares vehicle safety at a **4-way blind intersection** with and without **V2X (Vehicle-to-Everything)** communication. The simulation demonstrates how **DSRC-based V2V and V2I** communication detects dangers that physical sensors (camera/radar) miss — particularly at intersections with occluded sight lines.

Two identical scenarios are run:
1. **Sensor-Only Mode** — vehicles rely on camera/radar (80m range, 120° FOV, line-of-sight required)
2. **V2X Mode** — vehicles additionally use DSRC communication (300m range, 360°, non-line-of-sight)

Safety metrics are compared quantitatively to demonstrate V2X benefits.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **DSRC** | Dedicated Short-Range Communications (IEEE 802.11p, 5.9 GHz) — radio technology enabling V2X |
| **V2V** | Vehicle-to-Vehicle — cars broadcast BSM (Basic Safety Message) with position, speed, heading, brake status |
| **V2I** | Vehicle-to-Infrastructure — traffic lights broadcast SPaT (Signal Phase & Timing) with phase + countdown |
| **BSM** | Basic Safety Message (SAE J2735) — broadcast 10x/sec, contains vehicle state |
| **SPaT** | Signal Phase and Timing — traffic light tells vehicles current phase and when it changes |
| **TTC** | Time-to-Collision — primary safety metric: distance ÷ closing speed |
| **Multi-Agent** | Each vehicle is an independent agent with perception → decision → action loop |

## Architecture
<img width="1710" height="2330" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/dfe68c27-256a-4fda-bada-14b53bf3e62c" />
