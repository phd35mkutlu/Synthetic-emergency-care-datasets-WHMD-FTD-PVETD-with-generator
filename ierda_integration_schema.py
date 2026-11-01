"""
Integrated Emergency Response Data Architecture (IERDA)
Master Integration Schema for IERAD + Synthetic Datasets
Authors: M. Kutlu Sengul, Vahap Tecim
Dokuz Eylul University, Management Information Systems

This document defines how all datasets interconnect to form a comprehensive
emergency response simulation environment.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

# =============================================================================
# INTEGRATED DATA ARCHITECTURE OVERVIEW
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTEGRATED EMERGENCY RESPONSE DATA ARCHITECTURE           │
│                              (IERDA) v1.0                                   │
└─────────────────────────────────────────────────────────────────────────────┘

                          ┌─────────────────────┐
                          │   IERAD (Primary)   │
                          │  Sydney 2018-2024   │
                          │   Emergency Events  │
                          └──────────┬──────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
            ▼                        ▼                        ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │      WHMD       │    │       FTD       │    │      PVETD      │
   │    Wearable     │    │     Fleet       │    │ Private Vehicle │
   │ Health Monitor  │    │   Telemetry     │    │   Emergency     │
   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
            │                      │                      │
            │    ┌─────────────────┼─────────────────┐    │
            │    │                 │                 │    │
            ▼    ▼                 ▼                 ▼    ▼
   ┌───────────────────────────────────────────────────────────────┐
   │                    UNIFIED EVENT STREAM                        │
   │  • Timestamp-aligned (10-minute intervals)                    │
   │  • Geographic-aligned (Sydney metropolitan regions)           │
   │  • Incident-linked (via incident_id)                          │
   └───────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌───────────────────────────────────────────────────────────────┐
   │                    ANALYTICS & ML LAYER                        │
   │  • Risk Assessment          • Dispatch Optimization           │
   │  • Predictive Modeling      • Resource Allocation             │
   │  • Pattern Detection        • Response Time Prediction        │
   └───────────────────────────────────────────────────────────────┘
"""

# =============================================================================
# DATASET SPECIFICATIONS SUMMARY
# =============================================================================

DATASET_REGISTRY = {
    "IERAD": {
        "full_name": "Integrated Emergency Response Analytics Dataset",
        "source": "Metropolitan Sydney, Australia",
        "temporal_range": "2018-01-01 to 2024-12-31",
        "temporal_resolution": "10-minute intervals",
        "record_count": "~3.15M records",  # 6 years * 365 days * 144 intervals
        "features": 21,
        "primary_key": "timestamp + region_type",
        "synthetic": False,
        "role": "Primary incident/event dataset"
    },
    "WHMD": {
        "full_name": "Wearable Health Monitoring Dataset",
        "source": "Synthetic (aligned with IERAD)",
        "temporal_range": "2018-01-01 to 2024-12-31",
        "temporal_resolution": "10-minute intervals",
        "record_count": "~15.75M records",  # 5000 users * 3.15M intervals
        "features": 42,
        "primary_key": "user_id + timestamp",
        "foreign_key": "linked_incident_id → IERAD",
        "synthetic": True,
        "role": "Human health telemetry stream"
    },
    "FTD": {
        "full_name": "Fleet Telemetry Dataset",
        "source": "Synthetic (aligned with IERAD)",
        "temporal_range": "2018-01-01 to 2024-12-31",
        "temporal_resolution": "10-minute intervals",
        "record_count": "~630K records",  # 200 vehicles * 3.15M intervals (active only)
        "features": 58,
        "primary_key": "vehicle_id + timestamp",
        "foreign_key": "linked_incident_id → IERAD",
        "synthetic": True,
        "role": "Emergency fleet operational data"
    },
    "PVETD": {
        "full_name": "Private Vehicle Emergency Telemetry Dataset",
        "source": "Synthetic (aligned with IERAD)",
        "temporal_range": "2018-01-01 to 2024-12-31",
        "temporal_resolution": "10-minute intervals (routine), event-based (emergency)",
        "record_count": "~1.5M records",  # Subset of vehicles with events
        "features": 72,
        "primary_key": "vehicle_id + timestamp",
        "foreign_key": "linked_incident_id → IERAD (Accident type)",
        "synthetic": True,
        "role": "Connected car emergency detection"
    }
}

# =============================================================================
# TEMPORAL ALIGNMENT SPECIFICATION
# =============================================================================

TEMPORAL_ALIGNMENT = {
    "base_interval": "10 minutes",
    "timestamp_format": "ISO 8601",
    "timezone": "Australia/Sydney (AEST/AEDT)",
    "alignment_rule": "All datasets share identical timestamp intervals",
    
    "interval_calculation": {
        "intervals_per_day": 144,  # 24 * 6
        "intervals_per_year": 52560,
        "total_intervals_2018_2024": 315360  # 6 years * 52560
    },
    
    "synchronization_method": """
    Each dataset generates records at the same 10-minute boundaries.
    Timestamp = floor(actual_time / 10 minutes) * 10 minutes
    
    Example: 
    - Event at 14:23:45 → Timestamp: 14:20:00
    - Event at 14:27:12 → Timestamp: 14:20:00
    - Event at 14:31:00 → Timestamp: 14:30:00
    """
}

# =============================================================================
# GEOGRAPHIC ALIGNMENT SPECIFICATION
# =============================================================================

GEOGRAPHIC_ALIGNMENT = {
    "coverage_area": "Metropolitan Sydney, Australia",
    "bounding_box": {
        "north": -33.50,
        "south": -34.20,
        "west": 150.50,
        "east": 151.50
    },
    
    "region_types": {
        "Urban": {
            "description": "Sydney CBD and inner suburbs",
            "lat_range": (-33.85, -33.90),
            "lon_range": (151.18, 151.25),
            "population_density": "high",
            "typical_response_time": "5-10 minutes"
        },
        "Suburban": {
            "description": "Middle and outer suburbs",
            "lat_range": (-33.75, -34.00),
            "lon_range": (150.90, 151.30),
            "population_density": "medium",
            "typical_response_time": "8-15 minutes"
        },
        "Rural": {
            "description": "Fringe and semi-rural areas",
            "lat_range": (-33.50, -34.20),
            "lon_range": (150.50, 151.50),
            "population_density": "low",
            "typical_response_time": "15-30 minutes"
        }
    },
    
    "alignment_rule": "All datasets use consistent region_type classification"
}

# =============================================================================
# INCIDENT LINKAGE SPECIFICATION
# =============================================================================

class IncidentLinkageType(Enum):
    """Types of relationships between datasets and incidents"""
    CAUSED_BY = "Caused_By"          # Health event caused incident
    TRIGGERED = "Triggered"          # Vehicle crash triggered incident
    RESPONDED_TO = "Responded_To"    # Fleet responded to incident
    AFFECTED_BY = "Affected_By"      # Entity affected by incident
    PREDICTIVE = "Predictive"        # Data predicted incident

INCIDENT_LINKAGE = {
    "IERAD_to_WHMD": {
        "relationship": "One IERAD incident may link to multiple WHMD records",
        "linkage_scenarios": [
            {
                "scenario": "Cardiac arrest patient",
                "ierad_incident_type": "Cardiac Arrest",
                "whmd_records": "Pre-event vitals + event trigger + post-event monitoring"
            },
            {
                "scenario": "Fall victim",
                "ierad_incident_type": "Accident",
                "whmd_records": "Fall detection trigger + location + vitals"
            }
        ],
        "linkage_field": "linked_incident_id"
    },
    
    "IERAD_to_FTD": {
        "relationship": "One IERAD incident may have multiple FTD responses",
        "linkage_scenarios": [
            {
                "scenario": "Hybrid dispatch",
                "ierad_label": "Hybrid Dispatch",
                "ftd_records": "Drone + Ambulance telemetry during response"
            },
            {
                "scenario": "Multi-agency response",
                "ierad_incident_type": "Fire",
                "ftd_records": "Fire truck + Ambulance + Rescue vehicle"
            }
        ],
        "linkage_field": "linked_incident_id"
    },
    
    "IERAD_to_PVETD": {
        "relationship": "PVETD crash events create IERAD incidents",
        "linkage_scenarios": [
            {
                "scenario": "Vehicle accident detection",
                "pvetd_trigger": "Automatic crash detection",
                "ierad_incident_type": "Accident"
            }
        ],
        "linkage_field": "linked_incident_id",
        "direction": "PVETD → IERAD (incident creation)"
    }
}

# =============================================================================
# DATA FLOW ARCHITECTURE
# =============================================================================

DATA_FLOW_ARCHITECTURE = {
    "real_time_streams": {
        "WHMD": {
            "source": "Wearable devices via mobile apps",
            "protocol": "MQTT/HTTPS",
            "frequency": "Continuous (batched to 10-min)",
            "azure_ingestion": "Azure IoT Hub → Event Hubs"
        },
        "FTD": {
            "source": "Vehicle telematics systems",
            "protocol": "MQTT/CANbus gateway",
            "frequency": "Real-time (1-second) + 10-min aggregates",
            "azure_ingestion": "Azure IoT Hub → Event Hubs"
        },
        "PVETD": {
            "source": "Connected car platforms (eCall)",
            "protocol": "Cellular (4G/5G)",
            "frequency": "Event-triggered + periodic",
            "azure_ingestion": "Azure Functions → Event Hubs"
        }
    },
    
    "batch_integration": {
        "IERAD": {
            "source": "Historical emergency records",
            "format": "CSV/Parquet",
            "update_frequency": "Daily batch",
            "azure_storage": "Azure Data Lake Gen2"
        }
    },
    
    "processing_pipeline": {
        "step_1": "Data ingestion via Event Hubs",
        "step_2": "Stream processing via Azure Stream Analytics",
        "step_3": "ML inference via Azure ML",
        "step_4": "Storage in Cosmos DB (hot) / Data Lake (cold)",
        "step_5": "Analytics via Databricks",
        "step_6": "Visualization via Power BI"
    }
}

# =============================================================================
# USE CASE: CARDIAC ARREST SCENARIO DATA FLOW
# =============================================================================

CARDIAC_ARREST_SCENARIO = {
    "scenario_name": "Elderly Chronic Patient Cardiac Event",
    "description": """
    Demonstrates complete data flow from health monitoring detection
    through emergency response dispatch and hospital coordination.
    """,
    
    "timeline": [
        {
            "time": "T-30 min",
            "dataset": "WHMD",
            "event": "Routine vital monitoring",
            "data": {
                "heart_rate": 78,
                "blood_oxygen": 96,
                "activity": "Stationary",
                "risk_level": "Normal"
            }
        },
        {
            "time": "T-10 min",
            "dataset": "WHMD",
            "event": "Early warning signs",
            "data": {
                "heart_rate": 92,
                "blood_oxygen": 94,
                "heart_rate_variability": "decreasing",
                "risk_level": "Elevated"
            }
        },
        {
            "time": "T-0",
            "dataset": "WHMD",
            "event": "Emergency detection",
            "data": {
                "heart_rate": 142,
                "blood_oxygen": 88,
                "ecg_status": "AFib_Detected",
                "fall_detected": True,
                "alert_type": "Vital_Anomaly",
                "risk_level": "Critical",
                "emergency_probability": 0.94
            }
        },
        {
            "time": "T+0",
            "dataset": "IERAD",
            "event": "Incident created",
            "data": {
                "incident_type": "Cardiac Arrest",
                "incident_severity": "High",
                "emergency_level": "Critical",
                "region_type": "Suburban",
                "dispatch_coordinator": "AI"
            }
        },
        {
            "time": "T+1 min",
            "dataset": "FTD",
            "event": "Drone dispatched",
            "data": {
                "vehicle_type": "Emergency_Drone",
                "status": "En_Route",
                "payload_type": "AED",
                "eta_incident": "4 minutes"
            }
        },
        {
            "time": "T+1 min",
            "dataset": "FTD",
            "event": "Ambulance dispatched",
            "data": {
                "vehicle_type": "Ambulance",
                "status": "En_Route",
                "crew_count": 2,
                "eta_incident": "9 minutes"
            }
        },
        {
            "time": "T+4 min",
            "dataset": "FTD",
            "event": "Drone arrives",
            "data": {
                "vehicle_type": "Emergency_Drone",
                "status": "On_Scene",
                "defibrillator_status": "Deployed"
            }
        },
        {
            "time": "T+9 min",
            "dataset": "FTD",
            "event": "Ambulance arrives",
            "data": {
                "vehicle_type": "Ambulance",
                "status": "On_Scene",
                "response_time_current": 9.0
            }
        },
        {
            "time": "T+15 min",
            "dataset": "FTD",
            "event": "Patient transport",
            "data": {
                "vehicle_type": "Ambulance",
                "status": "Transporting",
                "patient_onboard": True,
                "patient_severity": "Critical",
                "patient_vitals_stable": True,
                "eta_hospital": "12 minutes"
            }
        },
        {
            "time": "T+15 min",
            "dataset": "WHMD",
            "event": "Continuous monitoring during transport",
            "data": {
                "heart_rate": 85,
                "blood_oxygen": 95,
                "connection_status": "Connected",
                "data_quality": "Good"
            }
        }
    ],
    
    "ierad_final_record": {
        "incident_type": "Cardiac Arrest",
        "incident_severity": "High",
        "emergency_level": "Critical",
        "region_type": "Suburban",
        "response_time": 9,  # Ambulance arrival
        "label": "Hybrid Dispatch",
        "dispatch_coordinator": "AI",
        "number_of_injuries": 1
    },
    
    "outcome_metrics": {
        "time_to_aed": "4 minutes",
        "time_to_paramedic": "9 minutes",
        "time_to_hospital": "27 minutes",
        "early_warning_benefit": "10 minutes advance detection",
        "survival_probability_increase": "estimated 15-20%"
    }
}

# =============================================================================
# VALIDATION FRAMEWORK
# =============================================================================

VALIDATION_FRAMEWORK = {
    "temporal_consistency": {
        "check": "All linked records share compatible timestamps",
        "tolerance": "±5 minutes for event alignment"
    },
    
    "geographic_consistency": {
        "check": "Incident location matches responding vehicle routes",
        "validation": "Distance calculations within region bounds"
    },
    
    "logical_consistency": {
        "checks": [
            "Response time ≥ dispatch time",
            "Vehicle speed within type limits",
            "Vital signs within physiological bounds",
            "Battery/fuel levels decrease during operation"
        ]
    },
    
    "statistical_validation": {
        "ierad_alignment": "Incident distribution matches IERAD patterns",
        "response_time_distribution": "Normal distribution around type averages",
        "user_type_distribution": "Reflects population demographics"
    }
}

# =============================================================================
# DATASET GENERATION PARAMETERS
# =============================================================================

GENERATION_PARAMETERS = {
    "WHMD": {
        "user_count": 5000,
        "user_type_distribution": {
            "Normal": 0.50,
            "Chronic_Patient": 0.15,
            "Elderly": 0.15,
            "Athlete": 0.10,
            "Swimmer": 0.05,
            "High_Risk_Worker": 0.05
        },
        "emergency_event_rate": 0.001,  # per user per day
        "data_quality_distribution": {
            "Good": 0.85,
            "Moderate": 0.12,
            "Poor": 0.03
        }
    },
    
    "FTD": {
        "fleet_composition": {
            "Ambulance": 50,
            "Emergency_Motorcycle": 20,
            "Fire_Truck": 30,
            "Rescue_Vehicle": 15,
            "Emergency_Drone": 40,
            "Autonomous_Robot": 10,
            "Emergency_Bicycle": 25
        },
        "availability_rate": 0.75,  # 75% available at any time
        "mission_duration_avg": 45  # minutes
    },
    
    "PVETD": {
        "connected_vehicle_count": 50000,
        "crash_rate_per_vehicle_year": 0.02,  # 2% annual crash rate
        "severity_distribution": {
            "Minor": 0.60,
            "Moderate": 0.25,
            "Severe": 0.12,
            "Critical": 0.03
        },
        "ecall_equipped_rate": 0.40  # 40% have automatic eCall
    }
}

# =============================================================================
# OUTPUT SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("INTEGRATED EMERGENCY RESPONSE DATA ARCHITECTURE (IERDA)")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("DATASET REGISTRY")
    print("-" * 80)
    for name, specs in DATASET_REGISTRY.items():
        print(f"\n{name}: {specs['full_name']}")
        print(f"  Features: {specs['features']}")
        print(f"  Records: {specs['record_count']}")
        print(f"  Role: {specs['role']}")
    
    print("\n" + "-" * 80)
    print("TEMPORAL ALIGNMENT")
    print("-" * 80)
    print(f"  Base Interval: {TEMPORAL_ALIGNMENT['base_interval']}")
    print(f"  Total Intervals (2018-2024): {TEMPORAL_ALIGNMENT['interval_calculation']['total_intervals_2018_2024']:,}")
    
    print("\n" + "-" * 80)
    print("CARDIAC ARREST SCENARIO SUMMARY")
    print("-" * 80)
    print(f"  Scenario: {CARDIAC_ARREST_SCENARIO['scenario_name']}")
    print(f"  Timeline Events: {len(CARDIAC_ARREST_SCENARIO['timeline'])}")
    print(f"  Key Outcome: {CARDIAC_ARREST_SCENARIO['outcome_metrics']['time_to_aed']} to AED delivery")
