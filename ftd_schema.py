"""
Fleet Telemetry Dataset (FTD) Schema
Synthetic dataset for emergency and rescue vehicle telemetry
Authors: M. Kutlu Sengul, Vahap Tecim
Dokuz Eylul University, Management Information Systems

This dataset extends IERAD's fleet capabilities with detailed telemetry
from various emergency response vehicle types including autonomous systems.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict

# =============================================================================
# VEHICLE TYPE DEFINITIONS
# =============================================================================

class VehicleType(Enum):
    """
    Emergency and rescue vehicle categories with distinct operational profiles.
    """
    AMBULANCE = "Ambulance"                    # Traditional emergency medical service
    EMERGENCY_MOTORCYCLE = "Emergency_Motorcycle"  # Rapid response, urban access
    FIRE_TRUCK = "Fire_Truck"                  # Fire and rescue operations
    RESCUE_VEHICLE = "Rescue_Vehicle"          # Technical rescue, extraction
    EMERGENCY_DRONE = "Emergency_Drone"        # Aerial medical delivery, reconnaissance
    AUTONOMOUS_ROBOT = "Autonomous_Robot"      # Ground-based autonomous delivery
    EMERGENCY_BICYCLE = "Emergency_Bicycle"    # Ultra-rapid urban response
    PRIVATE_VEHICLE = "Private_Vehicle"        # Connected cars with emergency features

class VehicleStatus(Enum):
    """Real-time operational status of vehicles"""
    AVAILABLE = "Available"
    EN_ROUTE = "En_Route"
    ON_SCENE = "On_Scene"
    TRANSPORTING = "Transporting"
    AT_HOSPITAL = "At_Hospital"
    MAINTENANCE = "Maintenance"
    OUT_OF_SERVICE = "Out_of_Service"
    CHARGING = "Charging"  # For electric/drone
    RETURNING = "Returning"

class MissionType(Enum):
    """Types of emergency missions"""
    MEDICAL_EMERGENCY = "Medical_Emergency"
    CARDIAC_ARREST = "Cardiac_Arrest"
    TRAUMA = "Trauma"
    FIRE_RESPONSE = "Fire_Response"
    RESCUE_OPERATION = "Rescue_Operation"
    EQUIPMENT_DELIVERY = "Equipment_Delivery"
    RECONNAISSANCE = "Reconnaissance"
    PATIENT_TRANSFER = "Patient_Transfer"
    STANDBY = "Standby"

class PowerSource(Enum):
    """Vehicle power/fuel types"""
    DIESEL = "Diesel"
    PETROL = "Petrol"
    ELECTRIC = "Electric"
    HYBRID = "Hybrid"
    BATTERY = "Battery"  # For drones/robots

# =============================================================================
# DATASET SCHEMA DEFINITION
# =============================================================================

@dataclass
class FTDRecord:
    """
    Single record in the Fleet Telemetry Dataset.
    10-minute intervals aligned with IERAD and WHMD.
    """
    # Identification
    record_id: str                      # Unique record identifier
    vehicle_id: str                     # Fleet vehicle identifier
    vehicle_type: VehicleType           # Category of vehicle
    vehicle_callsign: str               # Radio callsign (e.g., "AMB-12", "DRONE-05")
    
    # Temporal (aligned with IERAD)
    timestamp: datetime                 # 10-minute interval timestamp
    
    # Location & Movement
    latitude: float                     # GPS latitude
    longitude: float                    # GPS longitude
    altitude: float                     # Meters (especially for drones)
    heading: float                      # Degrees (0-360)
    speed: float                        # km/h
    acceleration: float                 # m/s²
    
    # Geographic Context
    region_type: str                    # Urban/Suburban/Rural
    road_type: Optional[str]            # Highway/Local/Unpaved (ground vehicles)
    airspace_zone: Optional[str]        # Restricted/Controlled/Open (drones)
    
    # Vehicle Status
    status: VehicleStatus               # Current operational status
    mission_type: Optional[MissionType] # Current mission if active
    mission_id: Optional[str]           # Reference to mission
    
    # Power/Fuel
    power_source: PowerSource           # Fuel type
    fuel_level: float                   # Percentage (0-100)
    estimated_range: float              # km remaining
    charging_status: Optional[str]      # For electric: Charging/Not_Charging
    
    # Vehicle Diagnostics
    engine_status: str                  # Normal/Warning/Critical
    engine_temperature: float           # Celsius
    oil_pressure: Optional[float]       # PSI (combustion engines)
    tire_pressure: Optional[Dict]       # PSI per tire
    brake_status: str                   # Good/Worn/Critical
    transmission_status: str            # Normal/Warning/Fault
    
    # Equipment Status (vehicle-specific)
    medical_equipment_status: Optional[str]  # Ambulance: Ready/In_Use/Depleted
    defibrillator_status: Optional[str]      # Ready/Deployed/Not_Available
    oxygen_level: Optional[float]            # Percentage remaining
    stretcher_status: Optional[str]          # Available/Occupied
    
    # Drone-Specific
    propeller_status: Optional[str]          # All_Good/Degraded/Failed
    payload_weight: Optional[float]          # kg
    payload_type: Optional[str]              # AED/Medicine/Blood/Equipment
    flight_mode: Optional[str]               # Manual/Autonomous/Return_Home
    obstacle_detection: Optional[str]        # Clear/Obstacle_Detected
    wind_speed_at_altitude: Optional[float]  # m/s
    
    # Robot-Specific
    navigation_mode: Optional[str]           # Autonomous/Remote_Controlled
    obstacle_avoidance_status: Optional[str] # Active/Inactive
    terrain_type: Optional[str]              # Paved/Unpaved/Indoor
    
    # Communication
    radio_status: str                   # Active/Degraded/Failed
    data_connection: str                # 5G/4G/3G/Satellite/None
    connection_latency: float           # milliseconds
    last_heartbeat: datetime            # Last successful data transmission
    
    # Crew Information (manned vehicles)
    crew_count: Optional[int]           # Number of crew members
    paramedic_onboard: Optional[bool]   # Qualified paramedic present
    driver_fatigue_level: Optional[str] # Low/Moderate/High/Critical
    shift_hours_remaining: Optional[float]  # Hours until shift end
    
    # Patient Information (when transporting)
    patient_onboard: bool               # Is patient being transported
    patient_count: Optional[int]        # Number of patients
    patient_severity: Optional[str]     # Minor/Moderate/Severe/Critical
    patient_vitals_stable: Optional[bool]  # Real-time assessment
    estimated_hospital_arrival: Optional[datetime]
    
    # Environmental Sensors
    external_temperature: float         # Celsius
    humidity: float                     # Percentage
    visibility: str                     # Good/Moderate/Poor
    road_conditions: Optional[str]      # Dry/Wet/Icy/Flooded
    
    # Performance Metrics
    response_time_current: Optional[float]  # Minutes since dispatch
    distance_to_incident: Optional[float]   # km
    distance_to_hospital: Optional[float]   # km
    eta_incident: Optional[datetime]        # Estimated arrival at scene
    eta_hospital: Optional[datetime]        # Estimated arrival at hospital
    
    # Safety Systems
    lights_sirens_active: bool          # Emergency lights/sirens on
    collision_warning: bool             # Collision system triggered
    lane_departure_warning: bool        # Lane departure detected
    abs_engaged: bool                   # Anti-lock brakes active
    
    # IERAD Linkage
    linked_incident_id: Optional[str]   # Reference to IERAD incident
    dispatch_coordinator: str           # Human/AI

# =============================================================================
# VEHICLE TYPE SPECIFICATIONS
# =============================================================================

VEHICLE_SPECIFICATIONS = {
    VehicleType.AMBULANCE: {
        "speed_range": (0, 120),         # km/h
        "avg_response_speed_urban": 45,
        "avg_response_speed_rural": 80,
        "fuel_capacity": 80,              # liters
        "fuel_consumption": 15,           # L/100km
        "crew_capacity": 4,
        "patient_capacity": 2,
        "equipment": ["stretcher", "defibrillator", "oxygen", "medications", "monitors"],
        "power_source": PowerSource.DIESEL,
        "typical_response_time_urban": 8,  # minutes
        "typical_response_time_rural": 15,
        "operational_cost_per_km": 2.50,   # AUD
        "ierad_speed_range": (20, 80)      # Matches IERAD Ambulance_Speed
    },
    
    VehicleType.EMERGENCY_MOTORCYCLE: {
        "speed_range": (0, 140),
        "avg_response_speed_urban": 60,    # Faster in traffic
        "avg_response_speed_rural": 100,
        "fuel_capacity": 20,
        "fuel_consumption": 5,
        "crew_capacity": 1,
        "patient_capacity": 0,
        "equipment": ["first_aid_kit", "aed", "basic_medications"],
        "power_source": PowerSource.PETROL,
        "typical_response_time_urban": 5,
        "typical_response_time_rural": 10,
        "operational_cost_per_km": 0.80,
        "advantages": ["traffic_navigation", "narrow_access", "rapid_response"]
    },
    
    VehicleType.FIRE_TRUCK: {
        "speed_range": (0, 100),
        "avg_response_speed_urban": 35,
        "avg_response_speed_rural": 60,
        "fuel_capacity": 200,
        "fuel_consumption": 40,
        "crew_capacity": 6,
        "patient_capacity": 0,
        "equipment": ["water_tank", "hoses", "ladders", "rescue_tools", "breathing_apparatus"],
        "water_capacity": 3000,            # liters
        "power_source": PowerSource.DIESEL,
        "typical_response_time_urban": 7,
        "typical_response_time_rural": 12,
        "operational_cost_per_km": 5.00
    },
    
    VehicleType.RESCUE_VEHICLE: {
        "speed_range": (0, 110),
        "avg_response_speed_urban": 40,
        "avg_response_speed_rural": 70,
        "fuel_capacity": 100,
        "fuel_consumption": 20,
        "crew_capacity": 4,
        "patient_capacity": 2,
        "equipment": ["jaws_of_life", "winch", "cutting_tools", "stretchers", "ropes"],
        "power_source": PowerSource.DIESEL,
        "typical_response_time_urban": 10,
        "typical_response_time_rural": 18,
        "operational_cost_per_km": 3.50
    },
    
    VehicleType.EMERGENCY_DRONE: {
        "speed_range": (0, 100),           # km/h
        "avg_response_speed": 80,          # Consistent, no traffic
        "battery_capacity": 100,           # percentage
        "flight_time_max": 45,             # minutes
        "range_max": 30,                   # km
        "payload_capacity": 10,            # kg (matches IERAD Payload_Weight)
        "equipment": ["aed", "blood_products", "medications", "camera", "speaker"],
        "power_source": PowerSource.BATTERY,
        "typical_response_time": 4,        # minutes (fastest)
        "operational_cost_per_km": 0.50,
        "ierad_speed_range": (30, 100),    # Matches IERAD Drone_Speed
        "ierad_battery_range": (30, 100),  # Matches IERAD Battery_Life
        "altitude_range": (10, 120),       # meters
        "advantages": ["no_traffic", "direct_route", "rapid_deployment", "aerial_view"]
    },
    
    VehicleType.AUTONOMOUS_ROBOT: {
        "speed_range": (0, 25),            # km/h (slower, ground-based)
        "avg_response_speed": 15,
        "battery_capacity": 100,
        "operation_time_max": 180,         # minutes
        "range_max": 20,                   # km
        "payload_capacity": 50,            # kg (larger than drone)
        "equipment": ["medical_supplies", "equipment_storage", "communication_hub"],
        "power_source": PowerSource.BATTERY,
        "typical_response_time": 15,
        "operational_cost_per_km": 0.30,
        "navigation_type": "autonomous",
        "terrain_capability": ["paved", "indoor", "light_unpaved"],
        "advantages": ["larger_payload", "no_human_risk", "continuous_operation"]
    },
    
    VehicleType.EMERGENCY_BICYCLE: {
        "speed_range": (0, 40),
        "avg_response_speed_urban": 25,
        "avg_response_speed_rural": 20,
        "battery_capacity": 100,           # E-bike assist
        "range_max": 80,                   # km with assist
        "crew_capacity": 1,
        "patient_capacity": 0,
        "equipment": ["first_aid_kit", "aed", "oxygen_portable"],
        "power_source": PowerSource.ELECTRIC,
        "typical_response_time_urban": 4,
        "typical_response_time_rural": 12,
        "operational_cost_per_km": 0.10,
        "advantages": ["pedestrian_areas", "congested_traffic", "events", "low_cost"]
    },
    
    VehicleType.PRIVATE_VEHICLE: {
        "speed_range": (0, 130),
        "avg_speed_urban": 35,
        "avg_speed_rural": 80,
        "fuel_capacity": 60,
        "crew_capacity": 5,
        "patient_capacity": 0,
        "equipment": ["ecall_system", "crash_sensors", "location_beacon"],
        "power_source": PowerSource.HYBRID,
        "emergency_features": ["automatic_crash_notification", "health_sync", "sos_button"],
        "data_shared": ["location", "crash_detection", "driver_health", "vehicle_diagnostics"]
    }
}

# =============================================================================
# DISPATCH DECISION MATRIX (Aligned with IERAD Labels)
# =============================================================================

DISPATCH_DECISION_FACTORS = {
    "Drone_Only": {
        "conditions": [
            "equipment_delivery",
            "reconnaissance_needed",
            "inaccessible_by_ground",
            "minor_injury_aed_required",
            "time_critical_medication"
        ],
        "max_payload": 10,  # kg
        "weather_restrictions": ["Stormy"],
        "range_limit": 30   # km
    },
    
    "Ambulance_Only": {
        "conditions": [
            "patient_transport_required",
            "multiple_casualties",
            "advanced_medical_care_needed",
            "heavy_equipment_required"
        ],
        "crew_required": True,
        "patient_capacity": 2
    },
    
    "Hybrid_Dispatch": {
        "conditions": [
            "cardiac_arrest",
            "time_critical_with_transport",
            "remote_location",
            "multiple_resource_types_needed"
        ],
        "typical_combination": ["Drone for AED", "Ambulance for transport"],
        "coordination_required": True
    }
}

# =============================================================================
# DATASET STATISTICS
# =============================================================================

DATASET_SPECIFICATIONS = {
    "name": "Fleet Telemetry Dataset (FTD)",
    "version": "1.0",
    "temporal_coverage": "2018-01-01 to 2024-12-31",
    "temporal_resolution": "10-minute intervals",
    "geographic_coverage": "Metropolitan Sydney, Australia",
    "total_features": 58,
    "vehicle_types": 8,
    "ierad_alignment": True,
    "synthetic": True,
    "purpose": "Emergency fleet management simulation and optimization",
    
    "feature_categories": {
        "identification": 4,
        "temporal": 1,
        "location_movement": 6,
        "geographic_context": 3,
        "vehicle_status": 4,
        "power_fuel": 4,
        "diagnostics": 6,
        "equipment_status": 5,
        "drone_specific": 7,
        "robot_specific": 3,
        "communication": 4,
        "crew_info": 4,
        "patient_info": 5,
        "environmental": 4,
        "performance_metrics": 5,
        "safety_systems": 4,
        "linkage": 2
    }
}

if __name__ == "__main__":
    print("=" * 70)
    print("FLEET TELEMETRY DATASET (FTD) SCHEMA")
    print("=" * 70)
    print(f"\nDataset: {DATASET_SPECIFICATIONS['name']}")
    print(f"Version: {DATASET_SPECIFICATIONS['version']}")
    print(f"Total Features: {DATASET_SPECIFICATIONS['total_features']}")
    print(f"Vehicle Types: {DATASET_SPECIFICATIONS['vehicle_types']}")
    
    print("\n" + "-" * 70)
    print("VEHICLE SPECIFICATIONS")
    print("-" * 70)
    for v_type, specs in VEHICLE_SPECIFICATIONS.items():
        print(f"\n{v_type.value}:")
        print(f"  Speed Range: {specs['speed_range']} km/h")
        print(f"  Power Source: {specs['power_source'].value}")
        if 'typical_response_time_urban' in specs:
            print(f"  Response Time (Urban): {specs['typical_response_time_urban']} min")
        elif 'typical_response_time' in specs:
            print(f"  Response Time: {specs['typical_response_time']} min")
        print(f"  Operational Cost: ${specs.get('operational_cost_per_km', 'N/A')}/km")
