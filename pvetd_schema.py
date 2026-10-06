"""
Private Vehicle Emergency Telemetry Dataset (PVETD) Schema
Synthetic dataset for connected private vehicles with emergency detection
Authors: M. Kutlu Sengul, Vahap Tecim
Dokuz Eylul University, Management Information Systems

This dataset captures telemetry from private vehicles equipped with
emergency detection systems (eCall, crash sensors, driver health monitoring).
Integrates with IERAD for accident-type incidents.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict

# =============================================================================
# ENUMERATION DEFINITIONS
# =============================================================================

class VehicleCategory(Enum):
    """Private vehicle categories"""
    SEDAN = "Sedan"
    SUV = "SUV"
    HATCHBACK = "Hatchback"
    TRUCK = "Truck"
    MOTORCYCLE = "Motorcycle"
    COMMERCIAL_VAN = "Commercial_Van"

class DriverStatus(Enum):
    """Driver health/alertness status"""
    ALERT = "Alert"
    DROWSY = "Drowsy"
    DISTRACTED = "Distracted"
    IMPAIRED = "Impaired"
    MEDICAL_EVENT = "Medical_Event"
    UNKNOWN = "Unknown"

class CrashSeverity(Enum):
    """Detected crash severity levels"""
    NONE = "None"
    MINOR = "Minor"           # Fender bender, no injuries likely
    MODERATE = "Moderate"     # Significant impact, possible injuries
    SEVERE = "Severe"         # Major collision, likely injuries
    CRITICAL = "Critical"     # Catastrophic, life-threatening

class EmergencyTrigger(Enum):
    """How emergency was detected/triggered"""
    NONE = "None"
    AUTOMATIC_CRASH = "Automatic_Crash"    # Sensor-detected collision
    MANUAL_SOS = "Manual_SOS"              # Driver pressed SOS
    DRIVER_HEALTH = "Driver_Health"        # Health monitoring alert
    VEHICLE_ROLLOVER = "Vehicle_Rollover"  # Rollover detected
    AIRBAG_DEPLOYMENT = "Airbag_Deployment"
    EXTERNAL_REPORT = "External_Report"    # Other vehicle/witness

# =============================================================================
# DATASET SCHEMA
# =============================================================================

@dataclass
class PVETDRecord:
    """
    Single record in the Private Vehicle Emergency Telemetry Dataset.
    Captures both routine telemetry and emergency events.
    """
    # Identification
    record_id: str                      # Unique record identifier
    vehicle_id: str                     # Anonymized vehicle ID
    vehicle_category: VehicleCategory   # Type of vehicle
    vehicle_year: int                   # Manufacturing year
    telematics_provider: str            # OEM/Aftermarket system
    
    # Temporal
    timestamp: datetime                 # 10-minute interval (routine) or event time
    
    # Location
    latitude: float
    longitude: float
    altitude: float                     # Meters
    heading: float                      # Degrees
    region_type: str                    # Urban/Suburban/Rural
    road_type: str                      # Highway/Local Road/Unpaved Road
    
    # Vehicle Dynamics
    speed: float                        # km/h
    acceleration_x: float               # m/s² (longitudinal)
    acceleration_y: float               # m/s² (lateral)
    acceleration_z: float               # m/s² (vertical)
    yaw_rate: float                     # degrees/second
    steering_angle: float               # degrees
    brake_pressure: float               # Percentage
    throttle_position: float            # Percentage
    
    # Vehicle Systems
    engine_rpm: int
    engine_temperature: float           # Celsius
    fuel_level: float                   # Percentage
    odometer: float                     # km
    tire_pressure_fl: float             # PSI
    tire_pressure_fr: float
    tire_pressure_rl: float
    tire_pressure_rr: float
    
    # Safety Systems Status
    abs_active: bool
    traction_control_active: bool
    stability_control_active: bool
    lane_departure_warning: bool
    forward_collision_warning: bool
    blind_spot_warning: bool
    airbags_deployed: List[str]         # List of deployed airbags
    seatbelt_status: Dict[str, bool]    # Position: fastened
    
    # Driver Monitoring (if equipped)
    driver_present: bool
    driver_status: DriverStatus
    driver_heart_rate: Optional[int]    # From steering wheel sensors
    driver_attention_score: Optional[float]  # 0-100
    drowsiness_level: Optional[str]     # Low/Moderate/High
    hands_on_wheel: Optional[bool]
    eyes_on_road: Optional[bool]
    
    # Occupancy
    occupant_count: int                 # Detected occupants
    child_seat_detected: bool
    pet_detected: bool
    
    # Environmental
    external_temperature: float         # Celsius
    rain_detected: bool
    headlights_on: bool
    wipers_active: bool
    visibility_estimate: str            # Good/Moderate/Poor
    
    # Connectivity
    cellular_signal: str                # Strong/Moderate/Weak/None
    gps_accuracy: float                 # Meters
    last_cloud_sync: datetime
    ecall_system_status: str            # Ready/Triggered/Disabled
    
    # Emergency Event Data (populated when emergency detected)
    emergency_detected: bool
    emergency_trigger: EmergencyTrigger
    crash_detected: bool
    crash_severity: CrashSeverity
    crash_delta_v: Optional[float]      # Change in velocity (m/s)
    crash_direction: Optional[str]      # Front/Rear/Left/Right/Rollover
    time_since_crash: Optional[float]   # Seconds since impact
    
    # Emergency Response
    ecall_transmitted: bool             # Automatic call made
    manual_sos_pressed: bool
    emergency_contact_notified: bool
    location_shared_with_ems: bool
    voice_connection_established: bool
    
    # Estimated Injuries (from crash analysis)
    estimated_injury_probability: float  # 0-1
    estimated_injury_severity: str       # None/Minor/Moderate/Severe
    recommended_response: str            # None/Check/Ambulance/Fire_Rescue
    
    # Post-Crash Vehicle Status
    vehicle_operational: Optional[bool]
    doors_openable: Optional[bool]
    fire_detected: Optional[bool]
    fuel_leak_detected: Optional[bool]
    battery_disconnect_activated: Optional[bool]
    hazard_lights_activated: Optional[bool]
    
    # IERAD Linkage
    linked_incident_id: Optional[str]   # For accident incidents
    contributes_to_traffic_data: bool   # Vehicle data informs traffic flow

# =============================================================================
# CRASH DETECTION PARAMETERS
# =============================================================================

CRASH_DETECTION_THRESHOLDS = {
    "acceleration_threshold": {
        "minor": 4.0,      # g-force for minor impact
        "moderate": 8.0,   # g-force for moderate impact
        "severe": 15.0,    # g-force for severe impact
        "critical": 25.0   # g-force for critical impact
    },
    "delta_v_threshold": {
        "minor": 8,        # km/h change
        "moderate": 16,
        "severe": 32,
        "critical": 50
    },
    "rollover_detection": {
        "roll_rate_threshold": 75,  # degrees/second
        "sustained_tilt": 45        # degrees from vertical
    },
    "injury_probability_factors": [
        "crash_severity",
        "seatbelt_status",
        "airbag_deployment",
        "occupant_position",
        "delta_v",
        "intrusion_estimate"
    ]
}

# =============================================================================
# DRIVER HEALTH MONITORING PARAMETERS
# =============================================================================

DRIVER_MONITORING_THRESHOLDS = {
    "heart_rate": {
        "normal_range": (50, 100),
        "tachycardia_threshold": 120,
        "bradycardia_threshold": 45,
        "cardiac_event_pattern": "sudden_irregular"
    },
    "drowsiness_indicators": {
        "blink_duration_threshold": 0.5,  # seconds
        "head_nod_frequency": 3,          # per minute
        "lane_deviation_pattern": True
    },
    "attention_score": {
        "alert": (80, 100),
        "slightly_distracted": (60, 79),
        "distracted": (40, 59),
        "severely_distracted": (0, 39)
    }
}

# =============================================================================
# eCALL SYSTEM SPECIFICATIONS (EU Standard)
# =============================================================================

ECALL_SPECIFICATIONS = {
    "standard": "EU eCall (ERA-GLONASS compatible)",
    "activation_modes": ["automatic", "manual"],
    "minimum_data_set": {
        "message_identifier": True,
        "activation_type": True,  # Manual/Automatic
        "test_call": True,
        "position_confidence": True,
        "vehicle_location": True,
        "vehicle_direction": True,
        "timestamp": True,
        "vehicle_identification": True,  # VIN
        "vehicle_class": True,
        "fuel_type": True,
        "occupant_count": True,
        "service_provider": True
    },
    "optional_data": {
        "crash_severity": True,
        "crash_direction": True,
        "airbag_deployment": True,
        "seatbelt_status": True,
        "rollover_detected": True
    },
    "transmission_time": 4,  # seconds max
    "retry_attempts": 3,
    "fallback": "voice_call"
}

# =============================================================================
# INTEGRATION WITH IERAD
# =============================================================================

IERAD_INTEGRATION_MAPPING = {
    "incident_type": {
        "Accident": ["crash_detected", "emergency_trigger"],
        "mapping": {
            "CrashSeverity.MINOR": "Low",
            "CrashSeverity.MODERATE": "Medium",
            "CrashSeverity.SEVERE": "High",
            "CrashSeverity.CRITICAL": "High"
        }
    },
    "traffic_congestion_contribution": {
        "crash_causes_congestion": True,
        "congestion_increase_factor": {
            "minor": 1.2,
            "moderate": 1.5,
            "severe": 2.0,
            "critical": 3.0
        }
    },
    "number_of_injuries_estimate": {
        "from": "occupant_count",
        "probability_model": "crash_severity_based"
    }
}

# =============================================================================
# DATASET STATISTICS
# =============================================================================

DATASET_SPECIFICATIONS = {
    "name": "Private Vehicle Emergency Telemetry Dataset (PVETD)",
    "version": "1.0",
    "temporal_coverage": "2018-01-01 to 2024-12-31",
    "temporal_resolution": "10-minute intervals (routine), event-based (emergency)",
    "geographic_coverage": "Metropolitan Sydney, Australia",
    "total_features": 72,
    "vehicle_categories": 6,
    "ierad_alignment": True,
    "synthetic": True,
    "purpose": "Connected vehicle emergency detection and response coordination",
    
    "feature_categories": {
        "identification": 5,
        "temporal": 1,
        "location": 6,
        "vehicle_dynamics": 8,
        "vehicle_systems": 8,
        "safety_systems": 8,
        "driver_monitoring": 8,
        "occupancy": 3,
        "environmental": 5,
        "connectivity": 4,
        "emergency_event": 7,
        "emergency_response": 5,
        "injury_estimation": 3,
        "post_crash": 6,
        "linkage": 2
    },
    
    "emergency_event_rate": {
        "routine_records_per_crash": 10000,  # Approximately 1 crash per 10,000 routine records
        "crash_severity_distribution": {
            "minor": 0.60,
            "moderate": 0.25,
            "severe": 0.12,
            "critical": 0.03
        }
    }
}

if __name__ == "__main__":
    print("=" * 70)
    print("PRIVATE VEHICLE EMERGENCY TELEMETRY DATASET (PVETD) SCHEMA")
    print("=" * 70)
    print(f"\nDataset: {DATASET_SPECIFICATIONS['name']}")
    print(f"Total Features: {DATASET_SPECIFICATIONS['total_features']}")
    print(f"Vehicle Categories: {DATASET_SPECIFICATIONS['vehicle_categories']}")
    
    print("\n" + "-" * 70)
    print("CRASH DETECTION THRESHOLDS")
    print("-" * 70)
    for severity, threshold in CRASH_DETECTION_THRESHOLDS["acceleration_threshold"].items():
        print(f"  {severity.title()}: {threshold}g")
    
    print("\n" + "-" * 70)
    print("eCALL SPECIFICATIONS")
    print("-" * 70)
    print(f"  Standard: {ECALL_SPECIFICATIONS['standard']}")
    print(f"  Transmission Time: {ECALL_SPECIFICATIONS['transmission_time']} seconds")
    print(f"  Minimum Data Fields: {len(ECALL_SPECIFICATIONS['minimum_data_set'])}")
