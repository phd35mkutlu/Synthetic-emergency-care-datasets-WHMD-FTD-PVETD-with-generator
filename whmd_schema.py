"""
Wearable Health Monitoring Dataset (WHMD) Schema
Synthetic dataset for cloud-native emergency fleet management research
Authors: M. Kutlu Sengul, Vahap Tecim
Dokuz Eylul University, Management Information Systems

This dataset is designed to integrate with IERAD (Integrated Emergency Response Analytics Dataset)
for comprehensive emergency response simulation and validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict

# =============================================================================
# USER TYPE DEFINITIONS
# =============================================================================

class UserType(Enum):
    """
    User categories for health monitoring with distinct risk profiles
    and monitoring requirements.
    """
    NORMAL = "Normal"                    # Standard health tracking
    CHRONIC_PATIENT = "Chronic_Patient"  # Continuous vital monitoring
    ELDERLY = "Elderly"                  # Fall detection, mobility tracking
    ATHLETE = "Athlete"                  # Performance metrics, overexertion
    SWIMMER = "Swimmer"                  # Drowning detection, water activity
    HIGH_RISK_WORKER = "High_Risk_Worker"  # Occupational hazard monitoring

class RiskLevel(Enum):
    """Real-time calculated risk levels based on vital signs and context"""
    NORMAL = "Normal"
    ELEVATED = "Elevated"
    HIGH = "High"
    CRITICAL = "Critical"

class AlertType(Enum):
    """Types of automated alerts triggered by the system"""
    NONE = "None"
    VITAL_ANOMALY = "Vital_Anomaly"
    FALL_DETECTED = "Fall_Detected"
    SOS_TRIGGERED = "SOS_Triggered"
    INACTIVITY = "Inactivity"
    GEOFENCE_BREACH = "Geofence_Breach"
    DROWNING_RISK = "Drowning_Risk"
    OVEREXERTION = "Overexertion"
    HAZARD_EXPOSURE = "Hazard_Exposure"

# =============================================================================
# DATASET SCHEMA DEFINITION
# =============================================================================

@dataclass
class WHMDRecord:
    """
    Single record in the Wearable Health Monitoring Dataset.
    Designed for 10-minute intervals to align with IERAD timestamps.
    """
    # Identification
    record_id: str                      # Unique record identifier
    user_id: str                        # Anonymized user identifier
    user_type: UserType                 # Category of user
    device_id: str                      # Wearable device identifier
    
    # Temporal (aligned with IERAD)
    timestamp: datetime                 # 10-minute interval timestamp
    
    # Location
    latitude: float                     # GPS latitude (-33.5 to -34.2 for Sydney)
    longitude: float                    # GPS longitude (150.5 to 151.5 for Sydney)
    location_accuracy: float            # GPS accuracy in meters
    altitude: float                     # Elevation in meters
    region_type: str                    # Urban/Suburban/Rural (matches IERAD)
    
    # Vital Signs - Core
    heart_rate: int                     # BPM (40-200 range)
    heart_rate_variability: float       # HRV in milliseconds
    blood_oxygen: float                 # SpO2 percentage (85-100)
    body_temperature: float             # Celsius (35.0-42.0)
    respiratory_rate: int               # Breaths per minute (8-40)
    
    # Vital Signs - Advanced (Samsung Galaxy Watch compatible)
    ecg_status: str                     # Normal/AFib_Detected/Inconclusive/Not_Measured
    blood_pressure_systolic: Optional[int]   # mmHg (80-200)
    blood_pressure_diastolic: Optional[int]  # mmHg (50-120)
    stress_level: int                   # 0-100 scale
    
    # Activity Metrics
    steps_count: int                    # Steps in current interval
    activity_type: str                  # Stationary/Walking/Running/Swimming/Cycling/Sleeping
    activity_intensity: str             # Low/Moderate/High/Extreme
    calories_burned: float              # kcal in interval
    
    # Sleep Metrics (when applicable)
    sleep_stage: Optional[str]          # Awake/Light/Deep/REM/None
    sleep_quality_score: Optional[int]  # 0-100
    
    # Environmental Context
    ambient_temperature: float          # Celsius
    ambient_humidity: float             # Percentage
    ambient_noise_level: float          # Decibels
    uv_index: Optional[float]           # 0-11+ scale
    
    # Safety & Alerts
    fall_detected: bool                 # Accelerometer-based fall detection
    sos_button_pressed: bool            # Manual emergency trigger
    inactivity_duration: int            # Minutes without movement
    geofence_status: str                # Inside/Outside/Breach
    
    # Device Status
    device_battery: float               # Percentage (0-100)
    connection_status: str              # Connected/Intermittent/Disconnected
    data_quality: str                   # Good/Moderate/Poor
    
    # Computed Fields
    risk_level: RiskLevel               # Calculated risk assessment
    alert_type: AlertType               # Triggered alert if any
    emergency_probability: float        # ML-predicted probability (0-1)
    
    # IERAD Linkage (for incidents)
    linked_incident_id: Optional[str]   # Reference to IERAD incident if applicable

# =============================================================================
# USER TYPE SPECIFIC PARAMETERS
# =============================================================================

USER_TYPE_PROFILES = {
    UserType.NORMAL: {
        "age_range": (18, 65),
        "baseline_heart_rate": (60, 80),
        "activity_distribution": {
            "Stationary": 0.50,
            "Walking": 0.30,
            "Running": 0.05,
            "Cycling": 0.05,
            "Swimming": 0.02,
            "Sleeping": 0.08
        },
        "chronic_conditions": [],
        "monitoring_frequency": "standard",  # 10-min intervals
        "alert_sensitivity": "normal",
        "emergency_base_probability": 0.001
    },
    
    UserType.CHRONIC_PATIENT: {
        "age_range": (40, 85),
        "baseline_heart_rate": (65, 90),
        "activity_distribution": {
            "Stationary": 0.65,
            "Walking": 0.25,
            "Running": 0.01,
            "Cycling": 0.01,
            "Swimming": 0.00,
            "Sleeping": 0.08
        },
        "chronic_conditions": ["diabetes", "hypertension", "heart_disease", "copd"],
        "monitoring_frequency": "intensive",  # 5-min intervals possible
        "alert_sensitivity": "high",
        "emergency_base_probability": 0.015
    },
    
    UserType.ELDERLY: {
        "age_range": (65, 95),
        "baseline_heart_rate": (55, 85),
        "activity_distribution": {
            "Stationary": 0.70,
            "Walking": 0.18,
            "Running": 0.00,
            "Cycling": 0.00,
            "Swimming": 0.02,
            "Sleeping": 0.10
        },
        "chronic_conditions": ["arthritis", "osteoporosis", "cognitive_decline"],
        "monitoring_frequency": "intensive",
        "alert_sensitivity": "very_high",
        "fall_risk_factor": 3.5,  # Multiplier for fall probability
        "emergency_base_probability": 0.012
    },
    
    UserType.ATHLETE: {
        "age_range": (16, 45),
        "baseline_heart_rate": (45, 65),  # Lower resting HR
        "activity_distribution": {
            "Stationary": 0.30,
            "Walking": 0.15,
            "Running": 0.25,
            "Cycling": 0.15,
            "Swimming": 0.10,
            "Sleeping": 0.05
        },
        "chronic_conditions": [],
        "monitoring_frequency": "performance",  # Higher during activity
        "alert_sensitivity": "sport_specific",
        "max_heart_rate_threshold": 0.95,  # % of max HR before alert
        "emergency_base_probability": 0.003
    },
    
    UserType.SWIMMER: {
        "age_range": (10, 70),
        "baseline_heart_rate": (55, 75),
        "activity_distribution": {
            "Stationary": 0.40,
            "Walking": 0.20,
            "Running": 0.05,
            "Cycling": 0.05,
            "Swimming": 0.25,  # Higher swimming frequency
            "Sleeping": 0.05
        },
        "chronic_conditions": [],
        "monitoring_frequency": "water_activity",
        "alert_sensitivity": "drowning_focused",
        "water_detection": True,
        "emergency_base_probability": 0.005
    },
    
    UserType.HIGH_RISK_WORKER: {
        "age_range": (18, 65),
        "baseline_heart_rate": (60, 80),
        "activity_distribution": {
            "Stationary": 0.20,
            "Walking": 0.40,
            "Running": 0.05,
            "Cycling": 0.05,
            "Swimming": 0.00,
            "Sleeping": 0.00  # Not during work hours
        },
        "chronic_conditions": [],
        "monitoring_frequency": "occupational",
        "alert_sensitivity": "hazard_focused",
        "lone_worker_protocol": True,
        "hazard_types": ["height", "confined_space", "chemical", "electrical"],
        "emergency_base_probability": 0.008
    }
}

# =============================================================================
# VITAL SIGNS THRESHOLDS FOR ALERT GENERATION
# =============================================================================

VITAL_THRESHOLDS = {
    "heart_rate": {
        "critical_low": 40,
        "low": 50,
        "normal_low": 60,
        "normal_high": 100,
        "high": 120,
        "critical_high": 150
    },
    "blood_oxygen": {
        "critical": 90,
        "low": 94,
        "normal": 95
    },
    "body_temperature": {
        "hypothermia_severe": 32.0,
        "hypothermia_mild": 35.0,
        "normal_low": 36.1,
        "normal_high": 37.2,
        "fever_mild": 38.0,
        "fever_high": 39.0,
        "critical_high": 40.0
    },
    "blood_pressure_systolic": {
        "hypotension": 90,
        "normal_low": 90,
        "normal_high": 120,
        "elevated": 130,
        "hypertension_stage1": 140,
        "hypertension_stage2": 180,
        "crisis": 180
    },
    "inactivity_duration": {
        "normal": 30,
        "concerning": 60,
        "critical": 120  # 2 hours without movement
    }
}

# =============================================================================
# SYDNEY METROPOLITAN REGION DEFINITIONS (Aligned with IERAD)
# =============================================================================

SYDNEY_REGIONS = {
    "Urban": {
        "lat_range": (-33.85, -33.90),
        "lon_range": (151.18, 151.25),
        "population_density": "high",
        "emergency_access": "excellent"
    },
    "Suburban": {
        "lat_range": (-33.75, -34.00),
        "lon_range": (150.90, 151.30),
        "population_density": "medium",
        "emergency_access": "good"
    },
    "Rural": {
        "lat_range": (-33.50, -34.20),
        "lon_range": (150.50, 151.50),
        "population_density": "low",
        "emergency_access": "limited"
    }
}

# =============================================================================
# DATASET STATISTICS SUMMARY
# =============================================================================

DATASET_SPECIFICATIONS = {
    "name": "Wearable Health Monitoring Dataset (WHMD)",
    "version": "1.0",
    "temporal_coverage": "2018-01-01 to 2024-12-31",
    "temporal_resolution": "10-minute intervals",
    "geographic_coverage": "Metropolitan Sydney, Australia",
    "total_features": 42,
    "user_types": 6,
    "ierad_alignment": True,
    "synthetic": True,
    "purpose": "Emergency response simulation and ML model training",
    
    "feature_categories": {
        "identification": 4,
        "temporal": 1,
        "location": 5,
        "vital_signs_core": 5,
        "vital_signs_advanced": 4,
        "activity_metrics": 4,
        "sleep_metrics": 2,
        "environmental": 4,
        "safety_alerts": 5,
        "device_status": 3,
        "computed": 4,
        "linkage": 1
    }
}

if __name__ == "__main__":
    print("=" * 70)
    print("WEARABLE HEALTH MONITORING DATASET (WHMD) SCHEMA")
    print("=" * 70)
    print(f"\nDataset: {DATASET_SPECIFICATIONS['name']}")
    print(f"Version: {DATASET_SPECIFICATIONS['version']}")
    print(f"Temporal Coverage: {DATASET_SPECIFICATIONS['temporal_coverage']}")
    print(f"Geographic Coverage: {DATASET_SPECIFICATIONS['geographic_coverage']}")
    print(f"Total Features: {DATASET_SPECIFICATIONS['total_features']}")
    print(f"User Types: {DATASET_SPECIFICATIONS['user_types']}")
    
    print("\n" + "-" * 70)
    print("USER TYPE PROFILES")
    print("-" * 70)
    for user_type, profile in USER_TYPE_PROFILES.items():
        print(f"\n{user_type.value}:")
        print(f"  Age Range: {profile['age_range']}")
        print(f"  Baseline HR: {profile['baseline_heart_rate']} BPM")
        print(f"  Emergency Base Probability: {profile['emergency_base_probability']}")
        print(f"  Monitoring Frequency: {profile['monitoring_frequency']}")
