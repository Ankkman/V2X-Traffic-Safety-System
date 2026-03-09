"""
=============================================================================
V2X TRAFFIC SAFETY SYSTEM SIMULATION — IMPROVED VERSION
=============================================================================
Author: Ankur Debnath
Project: V2X Traffic Safety System Simulation
Course : Wipro Automotive Elective

Description:
    Simulates a 4-way intersection with multiple vehicles approaching from
    different directions. Runs TWO scenarios:
      1. WITHOUT V2X — vehicles rely only on sensor-based detection
      2. WITH V2X    — vehicles also use DSRC-based V2V & V2I communication
    
    Compares safety metrics (collisions, near-misses, TTC, reaction distance)
    to demonstrate the safety benefits of V2X communication.

Key Concepts: DSRC, V2V, V2I, Multi-Agent Communication, TTC
Tools: Python, NumPy, Matplotlib

CHANGES FROM V1:
    - Scenario re-tuned so cross-road blind-intersection conflicts dominate
    - Reduced same-lane following issues that masked V2X advantages
    - Added following-distance logic so same-lane vehicles maintain gap
    - Clearer separation between sensor-only failures and V2X saves
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


# =============================================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# =============================================================================
# All simulation parameters in one place.
# If interviewer asks "why these numbers?" — they are based on realistic values
# from DSRC/SAE J2735 standards and typical urban driving scenarios.

class Config:
    """Central configuration for the entire simulation."""
    
    # --- Simulation Parameters ---
    DT = 0.1                        # Time step (100ms — matches DSRC BSM rate)
    TOTAL_TIME = 25.0               # Total simulation duration in seconds
    
    # --- Intersection Geometry ---
    ROAD_WIDTH = 8.0                # Road width in meters (standard 2-lane road)
    INTERSECTION_SIZE = 12.0        # Size of intersection box
    WORLD_RANGE = 200.0             # Rendering range
    
    # --- Vehicle Parameters ---
    VEHICLE_LENGTH = 4.5            # Car length (typical sedan)
    VEHICLE_WIDTH = 2.0             # Car width
    MAX_SPEED = 50.0 / 3.6          # 50 km/h → m/s (urban speed limit)
    MAX_BRAKE_DECEL = -8.0          # Maximum braking deceleration (m/s²)
    COMFORTABLE_DECEL = -3.0        # Comfortable braking (m/s²)
    COMFORTABLE_ACCEL = 2.5         # Normal acceleration (m/s²)
    
    # --- Sensor Parameters (NON-V2X) ---
    # Represent camera + radar capability
    SENSOR_RANGE = 80.0             # Detection range (meters)
    SENSOR_FOV = 120.0              # Field of view in degrees (front-facing)
    # KEY LIMITATION: Sensors can't see around corners! This is what V2X solves.
    
    # --- DSRC / V2X Parameters ---
    # Based on IEEE 802.11p / SAE J2735 standards
    DSRC_RANGE = 300.0              # Communication range (meters)
    DSRC_MSG_INTERVAL = 0.1         # BSM broadcast interval (10 Hz per J2735)
    # DSRC is omnidirectional (360°) and penetrates buildings
    
    # --- Traffic Light ---
    GREEN_DURATION = 8.0            # Green phase per direction
    YELLOW_DURATION = 2.0           # Yellow phase
    
    # --- Safety Thresholds ---
    TTC_DANGER = 3.0                # TTC below this = dangerous
    TTC_CRITICAL = 1.5              # TTC below this = critical / near-miss
    COLLISION_DISTANCE = 3.0        # Within this = collision
    SAFE_FOLLOWING_DIST = 15.0      # Minimum following distance (meters)
    
    # --- Output ---
    OUTPUT_DIR = "outputs"


# =============================================================================
# SECTION 2: DATA STRUCTURES
# =============================================================================

class Direction(Enum):
    """Which direction a vehicle approaches the intersection from."""
    NORTH = "North"     # Traveling southward (negative Y)
    SOUTH = "South"     # Traveling northward (positive Y)
    EAST = "East"       # Traveling westward (negative X)
    WEST = "West"       # Traveling eastward (positive X)


class VehicleState(Enum):
    """
    Vehicle behavioral state — mini-FSM.
    Analogous to Unity NPC states: Patrol → Alert → Engage
    Here: CRUISING → BRAKING → STOPPED → ACCELERATING
    """
    CRUISING = "Cruising"
    BRAKING = "Braking"
    STOPPED = "Stopped"
    ACCELERATING = "Accelerating"
    COLLISION = "Collision"


class TrafficLightPhase(Enum):
    GREEN = "Green"
    YELLOW = "Yellow"
    RED = "Red"


@dataclass
class BSM:
    """
    Basic Safety Message — core V2V message in DSRC (SAE J2735).
    Broadcast by every V2X vehicle 10 times per second.
    Contains position, velocity, heading, brake status.
    
    Think of it as: each Unity NPC broadcasting its transform + state.
    """
    vehicle_id: int
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    speed: float
    heading: float
    brake_status: bool


@dataclass
class SPaT:
    """
    Signal Phase and Timing — V2I message from traffic lights.
    Tells vehicles current phase AND when it will change.
    Non-V2X cars only see current color; V2X cars know timing.
    """
    ns_phase: TrafficLightPhase
    ew_phase: TrafficLightPhase
    time_to_change: float


@dataclass
class Vehicle:
    """
    Single vehicle agent — like a Unity GameObject with:
    - Transform → position, heading
    - Rigidbody → velocity, speed
    - AI Script → state (FSM)
    """
    id: int
    direction: Direction
    position: np.ndarray
    velocity: np.ndarray
    speed: float
    heading: float
    state: VehicleState
    v2x_enabled: bool = False
    has_collided: bool = False
    collision_time: float = -1.0
    
    # History for plotting
    position_history: List = field(default_factory=list)
    speed_history: List = field(default_factory=list)
    state_history: List = field(default_factory=list)
    ttc_history: List = field(default_factory=list)
    time_history: List = field(default_factory=list)
    
    # Reaction tracking
    first_reaction_distance: float = -1.0
    reacted: bool = False


@dataclass
class SafetyMetrics:
    """Aggregated safety metrics for one simulation run."""
    collision_count: int = 0
    near_miss_count: int = 0
    min_ttc: float = float('inf')
    avg_reaction_distance: float = 0.0
    avg_speed_at_conflict: float = 0.0
    total_hard_brakes: int = 0
    collision_speeds: List = field(default_factory=list)
    reaction_distances: List = field(default_factory=list)
    ttc_over_time: List = field(default_factory=list)
    time_stamps: List = field(default_factory=list)
    collision_events: List = field(default_factory=list)
    near_miss_events: List = field(default_factory=list)


# =============================================================================
# SECTION 3: TRAFFIC LIGHT CONTROLLER
# =============================================================================

class TrafficLightController:
    """
    Manages traffic light phases at the intersection.
    Cycle: NS Green → NS Yellow → EW Green → EW Yellow → repeat
    
    In V2X mode: broadcasts SPaT with phase + timing info.
    In non-V2X mode: vehicles only see current color.
    """
    
    def __init__(self):
        self.phase_sequence = [
            (TrafficLightPhase.GREEN, TrafficLightPhase.RED, Config.GREEN_DURATION),
            (TrafficLightPhase.YELLOW, TrafficLightPhase.RED, Config.YELLOW_DURATION),
            (TrafficLightPhase.RED, TrafficLightPhase.GREEN, Config.GREEN_DURATION),
            (TrafficLightPhase.RED, TrafficLightPhase.YELLOW, Config.YELLOW_DURATION),
        ]
        self.current_phase_idx = 0
        self.phase_timer = 0.0
    
    def update(self, dt: float):
        """Advance traffic light by dt seconds."""
        self.phase_timer += dt
        current_duration = self.phase_sequence[self.current_phase_idx][2]
        if self.phase_timer >= current_duration:
            self.phase_timer = 0.0
            self.current_phase_idx = (self.current_phase_idx + 1) % len(self.phase_sequence)
    
    def get_phase(self, direction: Direction) -> TrafficLightPhase:
        """Get current light phase for a given direction."""
        ns_phase, ew_phase, _ = self.phase_sequence[self.current_phase_idx]
        if direction in [Direction.NORTH, Direction.SOUTH]:
            return ns_phase
        return ew_phase
    
    def get_spat_message(self) -> SPaT:
        """Generate SPaT message (V2I broadcast)."""
        ns_phase, ew_phase, duration = self.phase_sequence[self.current_phase_idx]
        return SPaT(
            ns_phase=ns_phase,
            ew_phase=ew_phase,
            time_to_change=max(0, duration - self.phase_timer)
        )


# =============================================================================
# SECTION 4: V2X COMMUNICATION SYSTEM
# =============================================================================

class V2XCommunicationSystem:
    """
    Simulates DSRC-based V2X communication.
    
    Key differences from sensors:
    - Range: 300m vs 80m
    - Coverage: 360° vs 120° FOV
    - Occlusion: Goes through buildings vs needs line-of-sight
    - Information: Includes intent (brake status) vs only position
    
    Unity analogy: Global event system where nearby NPCs share state.
    """
    
    def __init__(self):
        self.message_buffer: List[BSM] = []
    
    def broadcast_bsm(self, vehicle: Vehicle, timestamp: float):
        """Vehicle broadcasts its Basic Safety Message."""
        bsm = BSM(
            vehicle_id=vehicle.id,
            timestamp=timestamp,
            position=vehicle.position.copy(),
            velocity=vehicle.velocity.copy(),
            speed=vehicle.speed,
            heading=vehicle.heading,
            brake_status=(vehicle.state in [VehicleState.BRAKING, VehicleState.STOPPED])
        )
        self.message_buffer.append(bsm)
    
    def receive_messages(self, vehicle: Vehicle) -> List[BSM]:
        """Receive BSMs from vehicles within DSRC range (300m, 360°, through buildings)."""
        received = []
        for bsm in self.message_buffer:
            if bsm.vehicle_id == vehicle.id:
                continue
            distance = np.linalg.norm(vehicle.position - bsm.position)
            if distance <= Config.DSRC_RANGE:
                received.append(bsm)
        return received
    
    def clear_buffer(self):
        self.message_buffer = []


# =============================================================================
# SECTION 5: SENSOR SIMULATION
# =============================================================================

class SensorSystem:
    """
    Simulates onboard sensors (camera + radar).
    
    Models 3 key limitations:
    1. Limited range (80m)
    2. Limited field of view (120° forward)
    3. Line-of-sight occlusion (buildings block view at corners)
    """
    
    @staticmethod
    def detect_vehicles(observer: Vehicle, all_vehicles: List[Vehicle]) -> List[Vehicle]:
        """Returns vehicles detectable by observer's sensors."""
        detected = []
        for target in all_vehicles:
            if target.id == observer.id or target.has_collided:
                continue
            
            distance = np.linalg.norm(observer.position - target.position)
            if distance > Config.SENSOR_RANGE:
                continue
            if not SensorSystem._is_in_fov(observer, target):
                continue
            if not SensorSystem._has_line_of_sight(observer, target):
                continue
            
            detected.append(target)
        return detected
    
    @staticmethod
    def _is_in_fov(observer: Vehicle, target: Vehicle) -> bool:
        """Check if target is within sensor field of view."""
        to_target = target.position - observer.position
        angle_to_target = np.degrees(np.arctan2(to_target[1], to_target[0]))
        angle_diff = abs(((angle_to_target - observer.heading) + 180) % 360 - 180)
        return angle_diff <= Config.SENSOR_FOV / 2
    
    @staticmethod
    def _has_line_of_sight(observer: Vehicle, target: Vehicle) -> bool:
        """
        Check if buildings at intersection corners block the view.
        
        THIS IS THE CRITICAL LIMITATION:
        When observer is on NS road and target is on EW road,
        and both are outside the intersection zone, corner buildings
        block the line of sight.
        
        V2X bypasses this because radio goes through buildings.
        """
        obs_pos = observer.position
        tgt_pos = target.position
        
        obs_on_ns = abs(obs_pos[0]) < Config.ROAD_WIDTH
        obs_on_ew = abs(obs_pos[1]) < Config.ROAD_WIDTH
        tgt_on_ns = abs(tgt_pos[0]) < Config.ROAD_WIDTH
        tgt_on_ew = abs(tgt_pos[1]) < Config.ROAD_WIDTH
        
        # On perpendicular roads?
        if (obs_on_ns and tgt_on_ew) or (obs_on_ew and tgt_on_ns):
            boundary = Config.INTERSECTION_SIZE / 2 + 5.0
            obs_outside = (abs(obs_pos[0]) > boundary or abs(obs_pos[1]) > boundary)
            tgt_outside = (abs(tgt_pos[0]) > boundary or abs(tgt_pos[1]) > boundary)
            if obs_outside and tgt_outside:
                return False  # Blocked by corner building!
        
        return True


# =============================================================================
# SECTION 6: VEHICLE DECISION MAKING
# =============================================================================

class VehicleDecisionModule:
    """
    Decision-making module — the vehicle's "brain".
    
    Two modes:
    1. Sensor-only: React only to what you physically see
    2. V2X-enabled: React to sensor data + DSRC messages
    """
    
    @staticmethod
    def calculate_ttc(vehicle: Vehicle, other_pos: np.ndarray,
                      other_vel: np.ndarray) -> float:
        """
        Calculate Time-to-Collision.
        TTC = distance / closing_speed
        Lower TTC = more dangerous
        """
        relative_pos = other_pos - vehicle.position
        distance = np.linalg.norm(relative_pos)
        
        if distance < 0.1:
            return 0.0
        
        direction = relative_pos / distance
        relative_vel = other_vel - vehicle.velocity
        closing_speed = -np.dot(relative_vel, direction)
        
        if closing_speed <= 0.5:
            return float('inf')
        
        return distance / closing_speed
    
    @staticmethod
    def _check_same_lane_following(vehicle: Vehicle, all_vehicles_or_bsms,
                                    is_bsm: bool = False) -> Tuple[float, Optional[int]]:
        """
        Check for same-lane following distance.
        Returns (distance_to_leader, leader_id).
        """
        min_dist = float('inf')
        leader_id = None
        
        items = all_vehicles_or_bsms
        for item in items:
            if is_bsm:
                other_pos = item.position
                other_id = item.vehicle_id
                other_vel = item.velocity
            else:
                if item.id == vehicle.id or item.has_collided:
                    continue
                other_pos = item.position
                other_id = item.id
                other_vel = item.velocity
            
            # Check if same direction of travel (dot product of velocities)
            if vehicle.speed < 0.5:
                continue
            
            to_other = other_pos - vehicle.position
            dist = np.linalg.norm(to_other)
            
            if dist < 1.0 or dist > 80.0:
                continue
            
            # Is the other vehicle AHEAD of us? (in our direction of travel)
            heading_rad = np.radians(vehicle.heading)
            forward = np.array([np.cos(heading_rad), np.sin(heading_rad)])
            dot = np.dot(to_other / dist, forward)
            
            if dot > 0.8:  # Roughly same direction, ahead of us
                # Check lateral offset (are they in our lane?)
                lateral = abs(np.cross(forward, to_other))
                if lateral < Config.ROAD_WIDTH * 0.8:
                    if dist < min_dist:
                        min_dist = dist
                        leader_id = other_id
        
        return min_dist, leader_id
    
    @staticmethod
    def decide_action_sensor_only(vehicle: Vehicle, detected_vehicles: List[Vehicle],
                                   all_vehicles: List[Vehicle],
                                   traffic_light: TrafficLightPhase,
                                   distance_to_intersection: float) -> Tuple[float, str]:
        """
        Decision logic for NON-V2X vehicle (sensor only).
        
        Can only react to:
        1. Vehicles it can physically SEE
        2. Traffic light color (not timing)
        """
        # Priority 0: Same-lane following (prevent rear-end collisions for both modes)
        follow_dist, leader_id = VehicleDecisionModule._check_same_lane_following(
            vehicle, all_vehicles, is_bsm=False
        )
        if follow_dist < Config.SAFE_FOLLOWING_DIST and leader_id is not None:
            # Maintain safe following distance
            decel = VehicleDecisionModule._decel_to_stop(vehicle.speed, follow_dist - 5.0)
            return (max(decel, Config.COMFORTABLE_DECEL),
                    f"Following V{leader_id} at {follow_dist:.0f}m")
        
        # Priority 1: Traffic light
        light_visible_distance = 60.0
        if distance_to_intersection < light_visible_distance:
            stop_line = distance_to_intersection - Config.INTERSECTION_SIZE / 2
            
            if traffic_light == TrafficLightPhase.RED and stop_line > 2.0:
                decel = VehicleDecisionModule._decel_to_stop(vehicle.speed, stop_line)
                return (max(decel, Config.MAX_BRAKE_DECEL), "Red light — stopping")
            
            elif traffic_light == TrafficLightPhase.RED and vehicle.speed > 0.3:
                return (Config.MAX_BRAKE_DECEL, "Red light — emergency stop")
            
            elif traffic_light == TrafficLightPhase.YELLOW and stop_line > 5.0:
                decel = VehicleDecisionModule._decel_to_stop(vehicle.speed, stop_line)
                return (max(decel, Config.COMFORTABLE_DECEL), "Yellow — stopping")
        
        # Priority 2: Collision risk from detected vehicles
        min_ttc = float('inf')
        danger_id = -1
        for other in detected_vehicles:
            ttc = VehicleDecisionModule.calculate_ttc(
                vehicle, other.position, other.velocity
            )
            if ttc < min_ttc:
                min_ttc = ttc
                danger_id = other.id
        
        if min_ttc < Config.TTC_CRITICAL:
            return (Config.MAX_BRAKE_DECEL,
                    f"EMERGENCY: TTC={min_ttc:.1f}s with V{danger_id}")
        elif min_ttc < Config.TTC_DANGER:
            return (Config.COMFORTABLE_DECEL,
                    f"Danger: TTC={min_ttc:.1f}s with V{danger_id}")
        
        # Priority 3: Stopped → proceed on green
        if vehicle.state == VehicleState.STOPPED:
            if traffic_light == TrafficLightPhase.GREEN and min_ttc > Config.TTC_DANGER:
                return (Config.COMFORTABLE_ACCEL, "Green — proceeding")
            return (0.0, "Waiting")
        
        # Priority 4: Cruise
        if vehicle.speed < Config.MAX_SPEED - 0.5:
            return (Config.COMFORTABLE_ACCEL, "Accelerating to cruise")
        return (0.0, "Cruising")
    
    @staticmethod
    def decide_action_v2x(vehicle: Vehicle, detected_vehicles: List[Vehicle],
                          all_vehicles: List[Vehicle],
                          received_bsms: List[BSM], spat: SPaT,
                          distance_to_intersection: float) -> Tuple[float, str]:
        """
        Decision logic for V2X-ENABLED vehicle.
        
        KEY ADVANTAGES over sensor-only:
        1. Knows about HIDDEN vehicles via BSM (300m, 360°, through walls)
        2. Knows WHEN light changes via SPaT → predictive braking
        3. Knows if others are BRAKING → earlier reaction
        """
        # Get own phase from SPaT
        if vehicle.direction in [Direction.NORTH, Direction.SOUTH]:
            my_phase = spat.ns_phase
        else:
            my_phase = spat.ew_phase
        
        # Priority 0: Same-lane following
        follow_dist, leader_id = VehicleDecisionModule._check_same_lane_following(
            vehicle, all_vehicles, is_bsm=False
        )
        if follow_dist < Config.SAFE_FOLLOWING_DIST and leader_id is not None:
            decel = VehicleDecisionModule._decel_to_stop(vehicle.speed, follow_dist - 5.0)
            return (max(decel, Config.COMFORTABLE_DECEL),
                    f"V2V: Following V{leader_id} at {follow_dist:.0f}m")
        
        # Priority 1: V2I — Predictive traffic light
        stop_line = distance_to_intersection - Config.INTERSECTION_SIZE / 2
        
        if stop_line > 2.0:
            if my_phase == TrafficLightPhase.GREEN and spat.time_to_change < 3.5:
                # V2X ADVANTAGE: knows light is ABOUT to change → proactive braking
                decel = VehicleDecisionModule._decel_to_stop(vehicle.speed, stop_line)
                return (max(decel, Config.COMFORTABLE_DECEL),
                        f"V2I: Light changes in {spat.time_to_change:.1f}s — preemptive braking")
            
            elif my_phase == TrafficLightPhase.RED:
                decel = VehicleDecisionModule._decel_to_stop(vehicle.speed, stop_line)
                return (max(decel, Config.MAX_BRAKE_DECEL), "V2I: Red — stopping")
            
            elif my_phase == TrafficLightPhase.YELLOW:
                decel = VehicleDecisionModule._decel_to_stop(vehicle.speed, stop_line)
                return (max(decel, Config.COMFORTABLE_DECEL), "V2I: Yellow — stopping")
        
        # Priority 2: V2V — check ALL known vehicles
        min_ttc = float('inf')
        danger_source = ""
        
        # From sensors
        for other in detected_vehicles:
            ttc = VehicleDecisionModule.calculate_ttc(
                vehicle, other.position, other.velocity
            )
            if ttc < min_ttc:
                min_ttc = ttc
                danger_source = f"sensor V{other.id}"
        
        # From V2V BSMs — includes HIDDEN vehicles!
        for bsm in received_bsms:
            ttc = VehicleDecisionModule.calculate_ttc(
                vehicle, bsm.position, bsm.velocity
            )
            if ttc < min_ttc:
                min_ttc = ttc
                danger_source = f"V2V V{bsm.vehicle_id}"
            
            # EXTRA: If other car is braking, react even earlier
            if bsm.brake_status and ttc < Config.TTC_DANGER * 1.5:
                return (Config.COMFORTABLE_DECEL,
                        f"V2V: V{bsm.vehicle_id} braking, TTC={ttc:.1f}s")
        
        if min_ttc < Config.TTC_CRITICAL:
            return (Config.MAX_BRAKE_DECEL,
                    f"EMERGENCY: TTC={min_ttc:.1f}s ({danger_source})")
        elif min_ttc < Config.TTC_DANGER:
            return (Config.COMFORTABLE_DECEL,
                    f"Caution: TTC={min_ttc:.1f}s ({danger_source})")
        
        # Priority 3: Proceed on green
        if vehicle.state == VehicleState.STOPPED:
            if my_phase == TrafficLightPhase.GREEN and min_ttc > Config.TTC_DANGER:
                return (Config.COMFORTABLE_ACCEL, "V2I: Green — proceeding")
            return (0.0, f"V2I: Waiting, green in {spat.time_to_change:.1f}s")
        
        # Priority 4: Cruise
        if vehicle.speed < Config.MAX_SPEED - 0.5:
            return (Config.COMFORTABLE_ACCEL, "Accelerating to cruise")
        return (0.0, "Cruising")
    
    @staticmethod
    def _decel_to_stop(current_speed: float, distance: float) -> float:
        """Calculate deceleration to stop within distance. Uses v² = u² + 2as."""
        if distance <= 0.5:
            return Config.MAX_BRAKE_DECEL
        if current_speed <= 0:
            return 0.0
        return max(-(current_speed ** 2) / (2 * distance), Config.MAX_BRAKE_DECEL)


# =============================================================================
# SECTION 7: SCENARIO SETUP (IMPROVED)
# =============================================================================

class ScenarioSetup:
    """
    Creates intersection scenario emphasizing blind-corner conflicts.
    
    KEY DESIGN: Vehicles on perpendicular roads arrive at intersection
    at nearly the same time during traffic light transition, creating
    a scenario where:
    - Sensor-only cars can't see the cross-traffic behind buildings
    - V2X cars detect cross-traffic via BSM well before line-of-sight
    """
    
    @staticmethod
    def create_vehicles(v2x_enabled: bool) -> List[Vehicle]:
        """
        6 vehicles across 4 directions. Carefully timed for conflict.
        
             V1 (North→South)
              ↓
              |
        V2 →  +  ← V3
              |
              ↑
         V0 (South→North)
        
        V4: second car from south (well behind V0)
        V5: second car from west (well behind V2)
        """
        vehicles = []
        
        # V0: From South heading North — has GREEN initially
        # Will reach intersection at ~8.6s (120m / 13.9 m/s)
        vehicles.append(Vehicle(
            id=0, direction=Direction.SOUTH,
            position=np.array([2.0, -120.0]),
            velocity=np.array([0.0, Config.MAX_SPEED]),
            speed=Config.MAX_SPEED, heading=90.0,
            state=VehicleState.CRUISING, v2x_enabled=v2x_enabled
        ))
        
        # V1: From North heading South — has GREEN initially  
        vehicles.append(Vehicle(
            id=1, direction=Direction.NORTH,
            position=np.array([-2.0, 100.0]),
            velocity=np.array([0.0, -Config.MAX_SPEED]),
            speed=Config.MAX_SPEED, heading=270.0,
            state=VehicleState.CRUISING, v2x_enabled=v2x_enabled
        ))
        
        # V2: From West heading East — has RED initially, gets GREEN later
        # CRITICAL CONFLICT VEHICLE: Timed to enter intersection during
        # the phase transition. Sensor-only V0 won't see V2 behind building.
        # V2X V0 will receive V2's BSM and know it's coming.
        vehicles.append(Vehicle(
            id=2, direction=Direction.WEST,
            position=np.array([-110.0, 2.0]),
            velocity=np.array([Config.MAX_SPEED * 0.95, 0.0]),
            speed=Config.MAX_SPEED * 0.95, heading=0.0,
            state=VehicleState.CRUISING, v2x_enabled=v2x_enabled
        ))
        
        # V3: From East heading West — has RED initially
        # Creates cross-traffic conflict with V1
        vehicles.append(Vehicle(
            id=3, direction=Direction.EAST,
            position=np.array([105.0, -2.0]),
            velocity=np.array([-Config.MAX_SPEED * 0.9, 0.0]),
            speed=Config.MAX_SPEED * 0.9, heading=180.0,
            state=VehicleState.CRUISING, v2x_enabled=v2x_enabled
        ))
        
        # V4: Second car from South — FAR behind V0 (no following collision)
        vehicles.append(Vehicle(
            id=4, direction=Direction.SOUTH,
            position=np.array([2.0, -185.0]),      # 65m behind V0 (safe gap)
            velocity=np.array([0.0, Config.MAX_SPEED]),
            speed=Config.MAX_SPEED, heading=90.0,
            state=VehicleState.CRUISING, v2x_enabled=v2x_enabled
        ))
        
        # V5: Second car from West — FAR behind V2 (no following collision)
        vehicles.append(Vehicle(
            id=5, direction=Direction.WEST,
            position=np.array([-180.0, 2.0]),       # 70m behind V2 (safe gap)
            velocity=np.array([Config.MAX_SPEED * 0.95, 0.0]),
            speed=Config.MAX_SPEED * 0.95, heading=0.0,
            state=VehicleState.CRUISING, v2x_enabled=v2x_enabled
        ))
        
        return vehicles


# =============================================================================
# SECTION 8: SIMULATION ENGINE
# =============================================================================

class SimulationEngine:
    """
    Main simulation loop — like Unity's Update() running manually.
    
    Each timestep:
    1. Update traffic light
    2. V2X: broadcast BSMs
    3. Per vehicle: perceive → decide → act
    4. Check collisions
    5. Record metrics
    """
    
    def __init__(self, v2x_enabled: bool):
        self.v2x_enabled = v2x_enabled
        self.vehicles = ScenarioSetup.create_vehicles(v2x_enabled)
        self.traffic_light = TrafficLightController()
        self.v2x_system = V2XCommunicationSystem() if v2x_enabled else None
        self.sensor_system = SensorSystem()
        self.metrics = SafetyMetrics()
        self.time = 0.0
        self.event_log: List[str] = []
        
        # Track near-miss events (deduplicated)
        self._near_miss_pairs = set()
    
    def run(self) -> SafetyMetrics:
        """Run complete simulation, return safety metrics."""
        num_steps = int(Config.TOTAL_TIME / Config.DT)
        mode = "V2X" if self.v2x_enabled else "Sensor-Only"
        
        print(f"\n{'='*60}")
        print(f"  RUNNING SIMULATION: {mode} MODE")
        print(f"  Vehicles: {len(self.vehicles)} | Duration: {Config.TOTAL_TIME}s")
        print(f"{'='*60}")
        
        for step in range(num_steps):
            self.time = step * Config.DT
            
            # 1. Update traffic light
            self.traffic_light.update(Config.DT)
            
            # 2. V2X broadcasts
            if self.v2x_enabled:
                self.v2x_system.clear_buffer()
                for v in self.vehicles:
                    if not v.has_collided:
                        self.v2x_system.broadcast_bsm(v, self.time)
            
            spat = self.traffic_light.get_spat_message()
            
            # 3. Per-vehicle loop
            for v in self.vehicles:
                if v.has_collided:
                    self._record_vehicle_state(v)
                    continue
                
                # PERCEIVE
                detected = self.sensor_system.detect_vehicles(v, self.vehicles)
                dist_to_intersection = np.linalg.norm(v.position)
                current_light = self.traffic_light.get_phase(v.direction)
                
                # DECIDE
                if self.v2x_enabled:
                    received_bsms = self.v2x_system.receive_messages(v)
                    accel, reason = VehicleDecisionModule.decide_action_v2x(
                        v, detected, self.vehicles, received_bsms, spat,
                        dist_to_intersection
                    )
                else:
                    accel, reason = VehicleDecisionModule.decide_action_sensor_only(
                        v, detected, self.vehicles, current_light,
                        dist_to_intersection
                    )
                
                # ACT
                self._update_vehicle_physics(v, accel)
                
                # Track reactions
                if accel < Config.COMFORTABLE_DECEL and not v.reacted:
                    v.reacted = True
                    v.first_reaction_distance = dist_to_intersection
                    self.metrics.reaction_distances.append(dist_to_intersection)
                
                if accel <= Config.MAX_BRAKE_DECEL * 0.8:
                    self.metrics.total_hard_brakes += 1
                
                self._record_vehicle_state(v)
            
            # 4. Check collisions
            self._check_collisions()
            
            # 5. Record global metrics
            self._record_global_metrics()
        
        self._finalize_metrics()
        return self.metrics
    
    def _update_vehicle_physics(self, vehicle: Vehicle, acceleration: float):
        """Update position/velocity using kinematics: v=v+a*dt, x=x+v*dt"""
        heading_rad = np.radians(vehicle.heading)
        direction = np.array([np.cos(heading_rad), np.sin(heading_rad)])
        
        new_speed = vehicle.speed + acceleration * Config.DT
        new_speed = max(0.0, min(new_speed, Config.MAX_SPEED * 1.1))
        
        vehicle.velocity = direction * new_speed
        vehicle.speed = new_speed
        vehicle.position = vehicle.position + vehicle.velocity * Config.DT
        
        # Update state (mini-FSM transition)
        if vehicle.has_collided:
            vehicle.state = VehicleState.COLLISION
        elif new_speed < 0.1:
            vehicle.state = VehicleState.STOPPED
        elif acceleration < -1.0:
            vehicle.state = VehicleState.BRAKING
        elif acceleration > 0.5:
            vehicle.state = VehicleState.ACCELERATING
        else:
            vehicle.state = VehicleState.CRUISING
    
    def _check_collisions(self):
        """Pairwise distance check — like manual OnCollisionEnter."""
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                v1, v2 = self.vehicles[i], self.vehicles[j]
                if v1.has_collided or v2.has_collided:
                    continue
                
                distance = np.linalg.norm(v1.position - v2.position)
                if distance < Config.COLLISION_DISTANCE:
                    v1.has_collided = v2.has_collided = True
                    v1.collision_time = v2.collision_time = self.time
                    v1.state = v2.state = VehicleState.COLLISION
                    
                    self.metrics.collision_count += 1
                    col_speed = (v1.speed + v2.speed) / 2
                    self.metrics.collision_speeds.append(col_speed)
                    self.metrics.collision_events.append(
                        (self.time, v1.id, v2.id, v1.position.copy(), col_speed)
                    )
                    
                    msg = (f"  ⚠ COLLISION at t={self.time:.1f}s: "
                           f"V{v1.id}({v1.direction.value}) ↔ V{v2.id}({v2.direction.value}) "
                           f"at ({v1.position[0]:.1f},{v1.position[1]:.1f}) "
                           f"| Speed: {col_speed*3.6:.1f} km/h")
                    self.event_log.append(msg)
                    print(msg)
    
    def _record_vehicle_state(self, vehicle: Vehicle):
        vehicle.position_history.append(vehicle.position.copy())
        vehicle.speed_history.append(vehicle.speed * 3.6)
        vehicle.state_history.append(vehicle.state)
        vehicle.time_history.append(self.time)
    
    def _record_global_metrics(self):
        """Record min TTC across all pairs + detect near-misses."""
        min_ttc_step = float('inf')
        
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                v1, v2 = self.vehicles[i], self.vehicles[j]
                if v1.has_collided or v2.has_collided:
                    continue
                
                ttc = VehicleDecisionModule.calculate_ttc(v1, v2.position, v2.velocity)
                distance = np.linalg.norm(v1.position - v2.position)
                
                if ttc < min_ttc_step:
                    min_ttc_step = ttc
                
                # Near-miss detection (deduplicated per pair)
                pair_key = (min(v1.id, v2.id), max(v1.id, v2.id))
                if (ttc < Config.TTC_CRITICAL and distance < Config.COLLISION_DISTANCE * 5
                        and pair_key not in self._near_miss_pairs):
                    self._near_miss_pairs.add(pair_key)
                    self.metrics.near_miss_count += 1
                    self.metrics.near_miss_events.append(
                        (self.time, v1.id, v2.id, ttc)
                    )
        
        if min_ttc_step < self.metrics.min_ttc:
            self.metrics.min_ttc = min_ttc_step
        
        self.metrics.ttc_over_time.append(min(min_ttc_step, 10.0))
        self.metrics.time_stamps.append(self.time)
    
    def _finalize_metrics(self):
        if self.metrics.reaction_distances:
            self.metrics.avg_reaction_distance = np.mean(self.metrics.reaction_distances)
        
        conflict_speeds = []
        for v in self.vehicles:
            for pos, spd in zip(v.position_history, v.speed_history):
                if np.linalg.norm(pos) < Config.INTERSECTION_SIZE * 1.5:
                    conflict_speeds.append(spd)
        if conflict_speeds:
            self.metrics.avg_speed_at_conflict = np.mean(conflict_speeds)


# =============================================================================
# SECTION 9: VISUALIZATION
# =============================================================================

class Visualizer:
    """Creates all matplotlib plots for the simulation report."""
    
    VEHICLE_COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
    
    STATE_COLORS = {
        VehicleState.CRUISING: '#4CAF50',
        VehicleState.BRAKING: '#FF9800',
        VehicleState.STOPPED: '#F44336',
        VehicleState.ACCELERATING: '#2196F3',
        VehicleState.COLLISION: '#000000',
    }
    
    @staticmethod
    def plot_intersection_layout(vehicles: List[Vehicle], title: str, filename: str):
        """Bird's eye view: roads, buildings, vehicle trajectories, collision markers."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        road_w = Config.ROAD_WIDTH
        
        # Roads
        ax.fill_between([-road_w, road_w], -150, 150, color='#E0E0E0', alpha=0.5)
        ax.fill_betweenx([-road_w, road_w], -150, 150, color='#E0E0E0', alpha=0.5)
        
        # Intersection box
        ax.add_patch(patches.Rectangle(
            (-road_w, -road_w), 2*road_w, 2*road_w,
            linewidth=2, edgecolor='#333', facecolor='#BDBDBD', alpha=0.7
        ))
        
        # Lane dividers
        for coords in [([0,0],[-150,-road_w]), ([0,0],[road_w,150]),
                       ([-150,-road_w],[0,0]), ([road_w,150],[0,0])]:
            ax.plot(*coords, 'w--', linewidth=1, alpha=0.7)
        
        # Corner buildings (these block sensor line-of-sight)
        bsize, boff = 30, road_w + 2
        for bx, by in [(boff,boff), (-boff-bsize,boff),
                       (boff,-boff-bsize), (-boff-bsize,-boff-bsize)]:
            ax.add_patch(patches.Rectangle(
                (bx,by), bsize, bsize,
                edgecolor='#666', facecolor='#A1887F', alpha=0.6
            ))
            ax.text(bx+bsize/2, by+bsize/2, 'Building',
                   ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        
        # Vehicle trajectories
        for v in vehicles:
            pos = np.array(v.position_history)
            if len(pos) == 0:
                continue
            color = Visualizer.VEHICLE_COLORS[v.id % 6]
            ax.plot(pos[:,0], pos[:,1], '-', color=color, linewidth=2, alpha=0.7,
                   label=f'V{v.id} ({v.direction.value})')
            ax.plot(pos[0,0], pos[0,1], 'o', color=color, markersize=10, zorder=5)
            ax.annotate(f'V{v.id}\nStart', (pos[0,0], pos[0,1]),
                       textcoords="offset points", xytext=(10,10),
                       fontsize=8, color=color, fontweight='bold')
            ax.plot(pos[-1,0], pos[-1,1], 's', color=color, markersize=10, zorder=5)
            
            if v.has_collided:
                ci = min(int(v.collision_time / Config.DT), len(pos)-1)
                ax.plot(pos[ci,0], pos[ci,1], 'X', color='red', markersize=20,
                       zorder=10, markeredgecolor='black', markeredgewidth=2)
        
        # Traffic light indicators
        ax.plot(road_w+2, road_w+2, 's', color='yellow', markersize=12,
               markeredgecolor='black', zorder=5)
        ax.annotate('TL-NS', (road_w+2, road_w+2), textcoords="offset points",
                   xytext=(8,8), fontsize=8)
        
        ax.set_xlim(-150, 150); ax.set_ylim(-150, 150)
        ax.set_xlabel('X Position (meters)'); ax.set_ylabel('Y Position (meters)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {filename}")
    
    @staticmethod
    def plot_simulation_results(vehicles: List[Vehicle], title: str, filename: str):
        """Speed profiles + FSM state timelines for all vehicles."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Speed Profile
        ax1 = axes[0]
        for v in vehicles:
            color = Visualizer.VEHICLE_COLORS[v.id % 6]
            ax1.plot(v.time_history, v.speed_history, '-', color=color,
                    linewidth=2, label=f'V{v.id} ({v.direction.value})')
            if v.has_collided:
                ci = min(int(v.collision_time / Config.DT), len(v.speed_history)-1)
                ax1.axvline(x=v.collision_time, color='red', linestyle='--', alpha=0.5)
                ax1.plot(v.collision_time, v.speed_history[ci], 'X', color='red',
                        markersize=15, zorder=10)
        
        ax1.axhline(y=Config.MAX_SPEED*3.6, color='gray', linestyle=':', alpha=0.5,
                   label='Speed Limit')
        ax1.set_ylabel('Speed (km/h)'); ax1.set_ylim(-2, 60)
        ax1.set_title(f'{title} — Speed Profile', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9, ncol=3); ax1.grid(True, alpha=0.3)
        
        # FSM State Timeline
        ax2 = axes[1]
        state_map = {VehicleState.CRUISING:4, VehicleState.ACCELERATING:3,
                     VehicleState.BRAKING:2, VehicleState.STOPPED:1, VehicleState.COLLISION:0}
        
        for v in vehicles:
            color = Visualizer.VEHICLE_COLORS[v.id % 6]
            vals = [state_map.get(s, 0) for s in v.state_history]
            ax2.plot(v.time_history, [sv + v.id*0.06 for sv in vals],
                    '-', color=color, linewidth=2.5, alpha=0.8, label=f'V{v.id}')
        
        ax2.set_yticks([0,1,2,3,4])
        ax2.set_yticklabels(['Collision','Stopped','Braking','Accelerating','Cruising'])
        ax2.set_xlabel('Time (seconds)'); ax2.set_ylabel('Vehicle State')
        ax2.set_title(f'{title} — FSM State Timeline', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9, ncol=3); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {filename}")
    
    @staticmethod
    def plot_ttc_comparison(m1: SafetyMetrics, m2: SafetyMetrics, filename: str):
        """TTC over time for both modes — shows V2X maintains higher safety margin."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        ax.plot(m1.time_stamps, m1.ttc_over_time, '-', color='#F44336',
               linewidth=2, alpha=0.8, label='Without V2X (Sensor Only)')
        ax.plot(m2.time_stamps, m2.ttc_over_time, '-', color='#4CAF50',
               linewidth=2, alpha=0.8, label='With V2X (DSRC)')
        
        ax.axhline(y=Config.TTC_DANGER, color='orange', linestyle='--', alpha=0.7,
                   label=f'Danger ({Config.TTC_DANGER}s)')
        ax.axhline(y=Config.TTC_CRITICAL, color='red', linestyle='--', alpha=0.7,
                   label=f'Critical ({Config.TTC_CRITICAL}s)')
        ax.fill_between(m1.time_stamps, 0, Config.TTC_CRITICAL, color='red', alpha=0.1)
        ax.fill_between(m1.time_stamps, Config.TTC_CRITICAL, Config.TTC_DANGER,
                        color='orange', alpha=0.1)
        
        ax.set_xlabel('Time (seconds)'); ax.set_ylabel('Minimum TTC (seconds)')
        ax.set_title('Time-to-Collision Comparison: V2X vs Sensor-Only',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {filename}")
    
    @staticmethod
    def plot_safety_metrics(m1: SafetyMetrics, m2: SafetyMetrics, filename: str):
        """Side-by-side bar chart of all safety metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Safety Metrics Comparison: V2X vs Sensor-Only',
                    fontsize=16, fontweight='bold', y=1.02)
        
        colors = ['#F44336', '#4CAF50']
        labels = ['Sensor Only', 'With V2X']
        
        def make_bar(ax, vals, title, ylabel, fmt=None, higher_better=False):
            bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=1.2)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(ylabel)
            for bar, val in zip(bars, vals):
                txt = fmt.format(val) if fmt else str(val)
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02*max(max(vals),1),
                       txt, ha='center', va='bottom', fontweight='bold', fontsize=13)
        
        # Collisions (lower better)
        make_bar(axes[0,0], [m1.collision_count, m2.collision_count],
                'Collisions (Lower=Safer)', 'Count')
        
        # Near-Misses (lower better)
        make_bar(axes[0,1], [m1.near_miss_count, m2.near_miss_count],
                'Near-Misses (Lower=Safer)', 'Count')
        
        # Min TTC (higher better)
        t1 = m1.min_ttc if m1.min_ttc < 100 else 0
        t2 = m2.min_ttc if m2.min_ttc < 100 else 0
        make_bar(axes[0,2], [t1, t2], 'Minimum TTC (Higher=Safer)', 'Seconds', '{:.2f}s')
        
        # Reaction Distance (higher = reacted earlier)
        make_bar(axes[1,0], [m1.avg_reaction_distance, m2.avg_reaction_distance],
                'Avg Reaction Distance (Higher=Earlier)', 'Meters', '{:.1f}m')
        
        # Hard Brakes (lower = smoother)
        make_bar(axes[1,1], [m1.total_hard_brakes, m2.total_hard_brakes],
                'Hard Brake Events (Lower=Smoother)', 'Count')
        
        # Speed at intersection (lower = safer)
        make_bar(axes[1,2], [m1.avg_speed_at_conflict, m2.avg_speed_at_conflict],
                'Avg Speed at Intersection (Lower=Safer)', 'km/h', '{:.1f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {filename}")
    
    @staticmethod
    def plot_combined_dashboard(v_no: List[Vehicle], v_yes: List[Vehicle],
                                 m_no: SafetyMetrics, m_yes: SafetyMetrics, filename: str):
        """Combined trajectory comparison + range visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        for ax, vehicles, title, bcolor in [
            (axes[0], v_no, 'WITHOUT V2X (Sensor Only)', '#F44336'),
            (axes[1], v_yes, 'WITH V2X (DSRC Enabled)', '#4CAF50'),
        ]:
            road_w = Config.ROAD_WIDTH
            ax.fill_between([-road_w, road_w], -130, 130, color='#E0E0E0', alpha=0.5)
            ax.fill_betweenx([-road_w, road_w], -130, 130, color='#E0E0E0', alpha=0.5)
            ax.add_patch(patches.Rectangle(
                (-road_w,-road_w), 2*road_w, 2*road_w,
                linewidth=2, edgecolor='#333', facecolor='#BDBDBD', alpha=0.7
            ))
            
            for v in vehicles:
                pos = np.array(v.position_history)
                if len(pos) == 0:
                    continue
                color = Visualizer.VEHICLE_COLORS[v.id % 6]
                ax.plot(pos[:,0], pos[:,1], '-', color=color, linewidth=2,
                       alpha=0.7, label=f'V{v.id}')
                ax.plot(pos[0,0], pos[0,1], 'o', color=color, markersize=8)
                
                if v.has_collided:
                    ci = min(int(v.collision_time / Config.DT), len(pos)-1)
                    ax.plot(pos[ci,0], pos[ci,1], 'X', color='red', markersize=18,
                           zorder=10, markeredgecolor='black', markeredgewidth=2)
            
            # Show range circles on V2X plot
            if 'V2X' in title and len(vehicles) > 0:
                mid = len(vehicles[0].position_history) // 4
                if mid > 0:
                    p = vehicles[0].position_history[mid]
                    ax.add_patch(plt.Circle(p, Config.DSRC_RANGE, fill=False,
                                           color='green', linestyle='--', linewidth=1.5,
                                           alpha=0.4, label='DSRC (300m)'))
                    ax.add_patch(plt.Circle(p, Config.SENSOR_RANGE, fill=False,
                                           color='red', linestyle=':', linewidth=1.5,
                                           alpha=0.6, label='Sensor (80m)'))
            
            for spine in ax.spines.values():
                spine.set_edgecolor(bcolor)
                spine.set_linewidth(3)
            
            ax.set_xlim(-140, 140); ax.set_ylim(-140, 140)
            ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)')
            ax.set_title(title, fontsize=13, fontweight='bold', color=bcolor)
            ax.legend(loc='upper left', fontsize=8)
            ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        
        plt.suptitle('V2X Traffic Safety: Trajectory Comparison',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {filename}")
    
    @staticmethod
    def plot_v2x_awareness_comparison(filename: str):
        """
        BONUS PLOT: Visual diagram showing sensor vs V2X perception zones.
        This is great for explaining in interviews.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        for ax, title, show_dsrc in [
            (axes[0], 'Sensor-Only Perception', False),
            (axes[1], 'V2X + Sensor Perception', True),
        ]:
            road_w = Config.ROAD_WIDTH
            ax.fill_between([-road_w, road_w], -100, 100, color='#E0E0E0', alpha=0.5)
            ax.fill_betweenx([-road_w, road_w], -100, 100, color='#E0E0E0', alpha=0.5)
            
            # Buildings
            bsize, boff = 25, road_w + 2
            for bx, by in [(boff,boff), (-boff-bsize,boff),
                           (boff,-boff-bsize), (-boff-bsize,-boff-bsize)]:
                ax.add_patch(patches.Rectangle(
                    (bx,by), bsize, bsize,
                    edgecolor='#666', facecolor='#A1887F', alpha=0.7
                ))
            
            # Our car (approaching from south)
            car_pos = np.array([2.0, -50.0])
            ax.plot(car_pos[0], car_pos[1], 's', color='blue', markersize=15, zorder=10)
            ax.annotate('OUR CAR', car_pos, textcoords="offset points",
                       xytext=(15, 0), fontsize=10, fontweight='bold', color='blue')
            
            # Sensor cone (limited FOV, blocked by buildings)
            sensor_arc = patches.Wedge(
                car_pos, Config.SENSOR_RANGE, 30, 150,
                facecolor='red', alpha=0.15, edgecolor='red', linestyle=':'
            )
            ax.add_patch(sensor_arc)
            ax.annotate(f'Sensor\n{Config.SENSOR_RANGE:.0f}m, {Config.SENSOR_FOV:.0f}°',
                       car_pos + np.array([0, 30]), fontsize=9, color='red',
                       ha='center', style='italic')
            
            # Hidden car (on perpendicular road, behind building)
            hidden_pos = np.array([-60.0, 2.0])
            ax.plot(hidden_pos[0], hidden_pos[1], 's', color='red', markersize=12, zorder=10)
            
            if show_dsrc:
                # DSRC range (360°, through buildings)
                dsrc_circle = plt.Circle(car_pos, Config.DSRC_RANGE, fill=False,
                                        color='green', linestyle='--', linewidth=2, alpha=0.5)
                ax.add_patch(dsrc_circle)
                ax.annotate(f'DSRC Range\n{Config.DSRC_RANGE:.0f}m, 360°',
                           car_pos + np.array([60, -40]), fontsize=10, color='green',
                           fontweight='bold')
                
                # Dashed line showing V2X "sees" hidden car
                ax.plot([car_pos[0], hidden_pos[0]], [car_pos[1], hidden_pos[1]],
                       '--', color='green', linewidth=2, alpha=0.7)
                ax.annotate('DETECTED via V2V!', hidden_pos,
                           textcoords="offset points", xytext=(-20, 15),
                           fontsize=10, color='green', fontweight='bold')
            else:
                # Red X — can't see
                ax.annotate('HIDDEN!\n(behind building)', hidden_pos,
                           textcoords="offset points", xytext=(-30, 15),
                           fontsize=10, color='red', fontweight='bold')
                ax.plot(hidden_pos[0], hidden_pos[1], 'X', color='darkred',
                       markersize=20, zorder=11)
            
            ax.set_xlim(-100, 100); ax.set_ylim(-100, 100)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)')
        
        plt.suptitle('Why V2X? — Sensor Limitations at Blind Intersections',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {filename}")


# =============================================================================
# SECTION 10: REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Formatted terminal report comparing both modes."""
    
    @staticmethod
    def generate(m1: SafetyMetrics, m2: SafetyMetrics,
                log1: List[str], log2: List[str]):
        
        print("\n")
        print("=" * 70)
        print("  V2X TRAFFIC SAFETY SYSTEM — SIMULATION REPORT")
        print("=" * 70)
        
        print(f"\n{'─'*70}")
        print("  SIMULATION PARAMETERS")
        print(f"{'─'*70}")
        print(f"  Duration:             {Config.TOTAL_TIME}s")
        print(f"  Time Step:            {Config.DT}s ({int(1/Config.DT)} Hz)")
        print(f"  Vehicles:             6 (4 directions)")
        print(f"  Scenario:             4-way blind intersection")
        print(f"  Sensor Range:         {Config.SENSOR_RANGE}m (FOV: {Config.SENSOR_FOV}°)")
        print(f"  DSRC Range:           {Config.DSRC_RANGE}m (360°, non-line-of-sight)")
        print(f"  Speed Limit:          {Config.MAX_SPEED*3.6:.0f} km/h")
        
        print(f"\n{'─'*70}")
        print("  SAFETY METRICS COMPARISON")
        print(f"{'─'*70}")
        print(f"  {'Metric':<35} {'Sensor Only':>12} {'With V2X':>12} {'Result':>12}")
        print(f"  {'─'*35} {'─'*12} {'─'*12} {'─'*12}")
        
        def row(name, v1, v2, fmt='d', lower_better=True):
            if fmt == 'd':
                s1, s2 = f"{v1:d}", f"{v2:d}"
            elif fmt == 'f2':
                s1, s2 = f"{v1:.2f}s", f"{v2:.2f}s"
            elif fmt == 'f1m':
                s1, s2 = f"{v1:.1f}m", f"{v2:.1f}m"
            elif fmt == 'f1k':
                s1, s2 = f"{v1:.1f}km/h", f"{v2:.1f}km/h"
            else:
                s1, s2 = str(v1), str(v2)
            
            if lower_better:
                if v2 < v1:
                    imp = "✓ Better"
                elif v2 == v1:
                    imp = "Same"
                else:
                    imp = "Worse"
            else:
                if v2 > v1:
                    imp = "✓ Better"
                elif v2 == v1:
                    imp = "Same"
                else:
                    imp = "Worse"
            
            print(f"  {name:<35} {s1:>12} {s2:>12} {imp:>12}")
        
        row('Collisions', m1.collision_count, m2.collision_count, 'd', True)
        row('Near-Misses', m1.near_miss_count, m2.near_miss_count, 'd', True)
        
        t1 = m1.min_ttc if m1.min_ttc < 100 else 0
        t2 = m2.min_ttc if m2.min_ttc < 100 else 0
        row('Minimum TTC', t1, t2, 'f2', False)  # Higher = better
        
        row('Avg Reaction Distance', m1.avg_reaction_distance,
            m2.avg_reaction_distance, 'f1m', False)  # Higher = better (reacted earlier)
        row('Hard Brake Events', m1.total_hard_brakes, m2.total_hard_brakes, 'd', True)
        row('Avg Speed at Intersection', m1.avg_speed_at_conflict,
            m2.avg_speed_at_conflict, 'f1k', True)
        
        for label, log in [("SENSOR ONLY", log1), ("WITH V2X", log2)]:
            print(f"\n{'─'*70}")
            print(f"  EVENT LOG — {label}")
            print(f"{'─'*70}")
            if log:
                for e in log:
                    print(e)
            else:
                print(f"  ✓ No collision or critical events!")
        
        print(f"\n{'─'*70}")
        print("  ANALYSIS")
        print(f"{'─'*70}")
        
        if m2.collision_count < m1.collision_count:
            pct = ((m1.collision_count - m2.collision_count) / max(m1.collision_count,1)) * 100
            print(f"  ✓ V2X reduced collisions by {pct:.0f}%"
                  f" ({m1.collision_count} → {m2.collision_count})")
        elif m2.collision_count == 0 and m1.collision_count == 0:
            print(f"  ✓ No collisions in either mode (safe scenario)")
        elif m2.collision_count == m1.collision_count:
            print(f"  ≈ Same collision count — but check other metrics")
        
        if m2.avg_reaction_distance > m1.avg_reaction_distance:
            diff = m2.avg_reaction_distance - m1.avg_reaction_distance
            print(f"  ✓ V2X reacted {diff:.1f}m earlier on average")
        
        if m2.total_hard_brakes < m1.total_hard_brakes:
            print(f"  ✓ V2X had fewer hard brakes ({m2.total_hard_brakes} vs {m1.total_hard_brakes})")
        
        if t2 > t1:
            print(f"  ✓ V2X maintained higher minimum TTC ({t2:.2f}s vs {t1:.2f}s)")
        
        print(f"\n  KEY V2X ADVANTAGES DEMONSTRATED:")
        print(f"  • DSRC range ({Config.DSRC_RANGE}m) >> Sensor range ({Config.SENSOR_RANGE}m)")
        print(f"  • 360° awareness vs {Config.SENSOR_FOV}° sensor FOV")
        print(f"  • Non-Line-of-Sight detection (through buildings at corners)")
        print(f"  • Predictive braking via SPaT (Signal Phase & Timing)")
        print(f"  • Cooperative awareness via BSM (brake status sharing)")
        
        print(f"\n{'='*70}")
        print("  END OF REPORT")
        print(f"{'='*70}\n")


# =============================================================================
# SECTION 11: MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  V2X TRAFFIC SAFETY SYSTEM SIMULATION")
    print("  Author: Ankur Debnath")
    print("  Tools: Python, NumPy, Matplotlib")
    print("  Key Concepts: DSRC, V2V/V2I, Multi-Agent Communication")
    print("="*70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Run 1: Without V2X
    sim1 = SimulationEngine(v2x_enabled=False)
    m1 = sim1.run()
    v1 = sim1.vehicles
    log1 = sim1.event_log
    
    # Run 2: With V2X
    sim2 = SimulationEngine(v2x_enabled=True)
    m2 = sim2.run()
    v2 = sim2.vehicles
    log2 = sim2.event_log
    
    # Generate visualizations
    print(f"\n{'─'*60}")
    print("  GENERATING VISUALIZATIONS...")
    print(f"{'─'*60}")
    
    Visualizer.plot_intersection_layout(v1, "Intersection — Without V2X (Sensor Only)",
                                        "01_intersection_no_v2x.png")
    Visualizer.plot_intersection_layout(v2, "Intersection — With V2X (DSRC Enabled)",
                                        "02_intersection_with_v2x.png")
    Visualizer.plot_simulation_results(v1, "Without V2X (Sensor Only)",
                                       "03_results_no_v2x.png")
    Visualizer.plot_simulation_results(v2, "With V2X (DSRC Enabled)",
                                       "04_results_with_v2x.png")
    Visualizer.plot_ttc_comparison(m1, m2, "05_ttc_comparison.png")
    Visualizer.plot_safety_metrics(m1, m2, "06_safety_metrics_comparison.png")
    Visualizer.plot_combined_dashboard(v1, v2, m1, m2, "07_combined_dashboard.png")
    Visualizer.plot_v2x_awareness_comparison("08_v2x_awareness_diagram.png")
    
    # Generate report
    ReportGenerator.generate(m1, m2, log1, log2)
    
    print("  All outputs saved to 'outputs/' directory.")
    print("  Simulation complete!\n")


if __name__ == "__main__":
    main()