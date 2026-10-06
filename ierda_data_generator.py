"""
IERDA - Integrated Emergency Response Data Architecture
Complete Data Generation System

This script generates synthetic datasets (WHMD, FTD, PVETD) aligned with IERAD
for cloud-native emergency fleet management research.

Authors: M. Kutlu Sengul, Vahap Tecim
Dokuz Eylul University, Management Information Systems

Usage:
    python ierda_data_generator.py --ierad_path <path_to_ierad_csv> --output_dir <output_directory>
    
    Or run directly - it will download IERAD from Kaggle automatically.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import argparse
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Global configuration parameters"""
    RANDOM_SEED = 42
    
    # User population for WHMD
    TOTAL_USERS = 1000  # Reduce for faster generation, increase for more data
    USER_TYPE_DISTRIBUTION = {
        "Normal": 0.50,
        "Chronic_Patient": 0.15,
        "Elderly": 0.15,
        "Athlete": 0.10,
        "Swimmer": 0.05,
        "High_Risk_Worker": 0.05
    }
    
    # Fleet composition for FTD
    FLEET_COMPOSITION = {
        "Ambulance": 30,
        "Emergency_Motorcycle": 15,
        "Fire_Truck": 20,
        "Rescue_Vehicle": 10,
        "Emergency_Drone": 25,
        "Autonomous_Robot": 5,
        "Emergency_Bicycle": 15
    }
    
    # Sydney geographic bounds
    SYDNEY_BOUNDS = {
        "lat_min": -34.2,
        "lat_max": -33.5,
        "lon_min": 150.5,
        "lon_max": 151.5
    }
    
    # Region centers
    REGION_CENTERS = {
        "Urban": {"lat": -33.87, "lon": 151.21},
        "Suburban": {"lat": -33.82, "lon": 151.00},
        "Rural": {"lat": -33.70, "lon": 150.70}
    }

# =============================================================================
# ENUMERATIONS
# =============================================================================

class UserType(Enum):
    NORMAL = "Normal"
    CHRONIC_PATIENT = "Chronic_Patient"
    ELDERLY = "Elderly"
    ATHLETE = "Athlete"
    SWIMMER = "Swimmer"
    HIGH_RISK_WORKER = "High_Risk_Worker"

class VehicleType(Enum):
    AMBULANCE = "Ambulance"
    EMERGENCY_MOTORCYCLE = "Emergency_Motorcycle"
    FIRE_TRUCK = "Fire_Truck"
    RESCUE_VEHICLE = "Rescue_Vehicle"
    EMERGENCY_DRONE = "Emergency_Drone"
    AUTONOMOUS_ROBOT = "Autonomous_Robot"
    EMERGENCY_BICYCLE = "Emergency_Bicycle"

class RiskLevel(Enum):
    NORMAL = "Normal"
    ELEVATED = "Elevated"
    HIGH = "High"
    CRITICAL = "Critical"

class VehicleStatus(Enum):
    AVAILABLE = "Available"
    EN_ROUTE = "En_Route"
    ON_SCENE = "On_Scene"
    TRANSPORTING = "Transporting"
    RETURNING = "Returning"

# =============================================================================
# USER PROFILES
# =============================================================================

USER_PROFILES = {
    "Normal": {
        "age_range": (18, 65),
        "baseline_hr": (60, 80),
        "baseline_spo2": (96, 100),
        "baseline_temp": (36.2, 37.0),
        "activity_weights": {"Stationary": 0.5, "Walking": 0.3, "Running": 0.1, "Sleeping": 0.1},
        "emergency_prob": 0.001,
        "fall_risk_multiplier": 1.0
    },
    "Chronic_Patient": {
        "age_range": (40, 85),
        "baseline_hr": (65, 95),
        "baseline_spo2": (92, 98),
        "baseline_temp": (36.0, 37.2),
        "activity_weights": {"Stationary": 0.65, "Walking": 0.25, "Running": 0.0, "Sleeping": 0.1},
        "emergency_prob": 0.015,
        "fall_risk_multiplier": 1.5,
        "conditions": ["diabetes", "hypertension", "heart_disease"]
    },
    "Elderly": {
        "age_range": (65, 95),
        "baseline_hr": (55, 85),
        "baseline_spo2": (93, 98),
        "baseline_temp": (35.8, 37.0),
        "activity_weights": {"Stationary": 0.7, "Walking": 0.18, "Running": 0.0, "Sleeping": 0.12},
        "emergency_prob": 0.012,
        "fall_risk_multiplier": 3.5
    },
    "Athlete": {
        "age_range": (16, 45),
        "baseline_hr": (45, 65),
        "baseline_spo2": (97, 100),
        "baseline_temp": (36.0, 37.5),
        "activity_weights": {"Stationary": 0.3, "Walking": 0.15, "Running": 0.35, "Sleeping": 0.2},
        "emergency_prob": 0.003,
        "fall_risk_multiplier": 0.8
    },
    "Swimmer": {
        "age_range": (10, 70),
        "baseline_hr": (55, 75),
        "baseline_spo2": (96, 100),
        "baseline_temp": (36.0, 37.2),
        "activity_weights": {"Stationary": 0.4, "Walking": 0.2, "Swimming": 0.3, "Sleeping": 0.1},
        "emergency_prob": 0.005,
        "fall_risk_multiplier": 1.0,
        "drowning_risk": True
    },
    "High_Risk_Worker": {
        "age_range": (18, 65),
        "baseline_hr": (60, 85),
        "baseline_spo2": (95, 99),
        "baseline_temp": (36.2, 37.5),
        "activity_weights": {"Stationary": 0.2, "Walking": 0.5, "Running": 0.1, "Working": 0.2},
        "emergency_prob": 0.008,
        "fall_risk_multiplier": 2.0,
        "hazard_types": ["height", "confined_space", "chemical"]
    }
}

# =============================================================================
# VEHICLE SPECIFICATIONS
# =============================================================================

VEHICLE_SPECS = {
    "Ambulance": {
        "speed_range": (20, 80),
        "response_time_urban": 8,
        "response_time_rural": 15,
        "fuel_capacity": 80,
        "crew_capacity": 4,
        "patient_capacity": 2,
        "cost_per_km": 2.50
    },
    "Emergency_Motorcycle": {
        "speed_range": (30, 120),
        "response_time_urban": 5,
        "response_time_rural": 10,
        "fuel_capacity": 20,
        "crew_capacity": 1,
        "patient_capacity": 0,
        "cost_per_km": 0.80
    },
    "Fire_Truck": {
        "speed_range": (20, 80),
        "response_time_urban": 7,
        "response_time_rural": 12,
        "fuel_capacity": 200,
        "crew_capacity": 6,
        "patient_capacity": 0,
        "cost_per_km": 5.00
    },
    "Rescue_Vehicle": {
        "speed_range": (20, 90),
        "response_time_urban": 10,
        "response_time_rural": 18,
        "fuel_capacity": 100,
        "crew_capacity": 4,
        "patient_capacity": 2,
        "cost_per_km": 3.50
    },
    "Emergency_Drone": {
        "speed_range": (30, 100),
        "response_time_urban": 4,
        "response_time_rural": 6,
        "battery_capacity": 100,
        "payload_capacity": 10,
        "flight_time_max": 45,
        "cost_per_km": 0.50
    },
    "Autonomous_Robot": {
        "speed_range": (5, 25),
        "response_time_urban": 15,
        "response_time_rural": 30,
        "battery_capacity": 100,
        "payload_capacity": 50,
        "cost_per_km": 0.30
    },
    "Emergency_Bicycle": {
        "speed_range": (15, 40),
        "response_time_urban": 4,
        "response_time_rural": 12,
        "battery_capacity": 100,
        "crew_capacity": 1,
        "cost_per_km": 0.10
    }
}

# =============================================================================
# DATA GENERATORS
# =============================================================================

class WHMDGenerator:
    """Wearable Health Monitoring Dataset Generator"""
    
    def __init__(self, seed: int = Config.RANDOM_SEED):
        np.random.seed(seed)
        self.users = self._create_user_base()
    
    def _create_user_base(self) -> pd.DataFrame:
        """Create base user population"""
        users = []
        user_id = 0
        
        for user_type, proportion in Config.USER_TYPE_DISTRIBUTION.items():
            count = int(Config.TOTAL_USERS * proportion)
            profile = USER_PROFILES[user_type]
            
            for _ in range(count):
                user_id += 1
                age = np.random.randint(profile["age_range"][0], profile["age_range"][1])
                
                users.append({
                    "user_id": f"USR_{user_id:05d}",
                    "user_type": user_type,
                    "age": age,
                    "baseline_hr_low": profile["baseline_hr"][0],
                    "baseline_hr_high": profile["baseline_hr"][1],
                    "baseline_spo2_low": profile["baseline_spo2"][0],
                    "baseline_spo2_high": profile["baseline_spo2"][1],
                    "emergency_prob": profile["emergency_prob"],
                    "fall_risk_multiplier": profile["fall_risk_multiplier"]
                })
        
        return pd.DataFrame(users)
    
    def generate_for_incident(self, incident_row: pd.Series, num_affected: int = 1) -> List[Dict]:
        """Generate WHMD records for a specific IERAD incident"""
        records = []
        
        # Select random users based on incident type
        if incident_row.get("Incident_Type") == "Cardiac Arrest":
            # Prefer chronic patients and elderly
            eligible_users = self.users[self.users["user_type"].isin(["Chronic_Patient", "Elderly", "Normal"])]
        else:
            eligible_users = self.users
        
        selected_users = eligible_users.sample(n=min(num_affected, len(eligible_users)))
        
        for _, user in selected_users.iterrows():
            # Generate location near incident
            region = incident_row.get("Region_Type", "Urban")
            center = Config.REGION_CENTERS.get(region, Config.REGION_CENTERS["Urban"])
            
            lat = center["lat"] + np.random.normal(0, 0.02)
            lon = center["lon"] + np.random.normal(0, 0.02)
            
            # Determine if this is an emergency situation
            is_emergency = incident_row.get("Incident_Severity") in ["High", "Medium"]
            
            # Generate vital signs
            if is_emergency:
                # Abnormal vitals for emergency
                hr = np.random.randint(100, 160)
                spo2 = np.random.randint(85, 94)
                temp = np.random.uniform(35.0, 39.5)
                risk_level = "Critical" if incident_row.get("Incident_Severity") == "High" else "High"
                emergency_prob = np.random.uniform(0.7, 0.95)
            else:
                # Normal vitals
                hr = np.random.randint(user["baseline_hr_low"], user["baseline_hr_high"])
                spo2 = np.random.randint(user["baseline_spo2_low"], user["baseline_spo2_high"])
                temp = np.random.uniform(36.0, 37.2)
                risk_level = "Normal"
                emergency_prob = np.random.uniform(0.01, 0.1)
            
            # Determine alerts
            fall_detected = (incident_row.get("Incident_Type") == "Accident" and 
                           np.random.random() < 0.3 * user["fall_risk_multiplier"])
            sos_pressed = is_emergency and np.random.random() < 0.4
            
            record = {
                "record_id": f"WHMD_{np.random.randint(100000, 999999)}",
                "user_id": user["user_id"],
                "user_type": user["user_type"],
                "timestamp": incident_row.get("Timestamp", datetime.now()),
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "region_type": region,
                "heart_rate": hr,
                "heart_rate_variability": round(np.random.uniform(20, 80), 1),
                "blood_oxygen": spo2,
                "body_temperature": round(temp, 1),
                "respiratory_rate": np.random.randint(12, 25) if not is_emergency else np.random.randint(20, 35),
                "ecg_status": "AFib_Detected" if (is_emergency and np.random.random() < 0.3) else "Normal",
                "blood_pressure_systolic": np.random.randint(90, 180) if is_emergency else np.random.randint(110, 130),
                "blood_pressure_diastolic": np.random.randint(60, 110) if is_emergency else np.random.randint(70, 85),
                "stress_level": np.random.randint(60, 100) if is_emergency else np.random.randint(10, 40),
                "steps_count": np.random.randint(0, 500),
                "activity_type": "Stationary" if is_emergency else np.random.choice(["Walking", "Stationary", "Running"]),
                "fall_detected": fall_detected,
                "sos_button_pressed": sos_pressed,
                "inactivity_duration": np.random.randint(30, 120) if is_emergency else 0,
                "device_battery": np.random.randint(20, 100),
                "connection_status": "Connected",
                "risk_level": risk_level,
                "alert_type": "Vital_Anomaly" if is_emergency else ("Fall_Detected" if fall_detected else "None"),
                "emergency_probability": round(emergency_prob, 3),
                "linked_incident_id": incident_row.name if hasattr(incident_row, 'name') else None
            }
            records.append(record)
        
        return records
    
    def generate_routine_records(self, timestamp: datetime, sample_size: int = 100) -> List[Dict]:
        """Generate routine (non-emergency) monitoring records"""
        records = []
        sampled_users = self.users.sample(n=min(sample_size, len(self.users)))
        
        for _, user in sampled_users.iterrows():
            region = np.random.choice(["Urban", "Suburban", "Rural"], p=[0.5, 0.35, 0.15])
            center = Config.REGION_CENTERS[region]
            
            record = {
                "record_id": f"WHMD_{np.random.randint(100000, 999999)}",
                "user_id": user["user_id"],
                "user_type": user["user_type"],
                "timestamp": timestamp,
                "latitude": round(center["lat"] + np.random.normal(0, 0.05), 6),
                "longitude": round(center["lon"] + np.random.normal(0, 0.05), 6),
                "region_type": region,
                "heart_rate": np.random.randint(user["baseline_hr_low"], user["baseline_hr_high"]),
                "heart_rate_variability": round(np.random.uniform(30, 70), 1),
                "blood_oxygen": np.random.randint(user["baseline_spo2_low"], user["baseline_spo2_high"]),
                "body_temperature": round(np.random.uniform(36.2, 37.0), 1),
                "respiratory_rate": np.random.randint(12, 18),
                "ecg_status": "Normal",
                "blood_pressure_systolic": np.random.randint(110, 130),
                "blood_pressure_diastolic": np.random.randint(70, 85),
                "stress_level": np.random.randint(10, 40),
                "steps_count": np.random.randint(0, 2000),
                "activity_type": np.random.choice(list(USER_PROFILES[user["user_type"]]["activity_weights"].keys())),
                "fall_detected": False,
                "sos_button_pressed": False,
                "inactivity_duration": 0,
                "device_battery": np.random.randint(30, 100),
                "connection_status": np.random.choice(["Connected", "Connected", "Connected", "Intermittent"]),
                "risk_level": "Normal",
                "alert_type": "None",
                "emergency_probability": round(np.random.uniform(0.001, 0.05), 4),
                "linked_incident_id": None
            }
            records.append(record)
        
        return records


class FTDGenerator:
    """Fleet Telemetry Dataset Generator"""
    
    def __init__(self, seed: int = Config.RANDOM_SEED):
        np.random.seed(seed)
        self.fleet = self._create_fleet()
    
    def _create_fleet(self) -> pd.DataFrame:
        """Create base fleet"""
        vehicles = []
        vehicle_id = 0
        
        for v_type, count in Config.FLEET_COMPOSITION.items():
            specs = VEHICLE_SPECS[v_type]
            
            for i in range(count):
                vehicle_id += 1
                vehicles.append({
                    "vehicle_id": f"VEH_{vehicle_id:04d}",
                    "vehicle_type": v_type,
                    "callsign": f"{v_type[:3].upper()}-{i+1:02d}",
                    "speed_min": specs["speed_range"][0],
                    "speed_max": specs["speed_range"][1],
                    "response_time_urban": specs.get("response_time_urban", 10),
                    "response_time_rural": specs.get("response_time_rural", 20),
                    "cost_per_km": specs["cost_per_km"]
                })
        
        return pd.DataFrame(vehicles)
    
    def generate_for_incident(self, incident_row: pd.Series) -> List[Dict]:
        """Generate FTD records for responding to an incident"""
        records = []
        dispatch_label = incident_row.get("Label", "Ambulance Only")
        incident_type = incident_row.get("Incident_Type", "Other")
        region = incident_row.get("Region_Type", "Urban")
        
        # Determine which vehicles to dispatch
        if dispatch_label == "Drone Only":
            vehicle_types = ["Emergency_Drone"]
        elif dispatch_label == "Ambulance Only":
            vehicle_types = ["Ambulance"]
        else:  # Hybrid
            vehicle_types = ["Emergency_Drone", "Ambulance"]
        
        # Add fire truck for fire incidents
        if incident_type == "Fire":
            vehicle_types.append("Fire_Truck")
        
        # Get incident location
        center = Config.REGION_CENTERS.get(region, Config.REGION_CENTERS["Urban"])
        incident_lat = center["lat"] + np.random.normal(0, 0.02)
        incident_lon = center["lon"] + np.random.normal(0, 0.02)
        
        for v_type in vehicle_types:
            available_vehicles = self.fleet[self.fleet["vehicle_type"] == v_type]
            if len(available_vehicles) == 0:
                continue
            
            vehicle = available_vehicles.sample(n=1).iloc[0]
            specs = VEHICLE_SPECS[v_type]
            
            # Calculate response metrics
            distance = incident_row.get("Distance_to_Incident_km", np.random.uniform(2, 20))
            response_time = vehicle["response_time_urban"] if region == "Urban" else vehicle["response_time_rural"]
            
            # Determine vehicle status progression
            statuses = ["En_Route", "On_Scene"]
            if v_type == "Ambulance" and incident_row.get("Incident_Severity") in ["High", "Medium"]:
                statuses.append("Transporting")
            
            for status in statuses:
                speed = np.random.randint(vehicle["speed_min"], vehicle["speed_max"])
                
                record = {
                    "record_id": f"FTD_{np.random.randint(100000, 999999)}",
                    "vehicle_id": vehicle["vehicle_id"],
                    "vehicle_type": v_type,
                    "callsign": vehicle["callsign"],
                    "timestamp": incident_row.get("Timestamp", datetime.now()),
                    "latitude": round(incident_lat + np.random.normal(0, 0.01), 6),
                    "longitude": round(incident_lon + np.random.normal(0, 0.01), 6),
                    "altitude": np.random.randint(0, 120) if v_type == "Emergency_Drone" else 0,
                    "heading": np.random.randint(0, 360),
                    "speed": speed,
                    "region_type": region,
                    "status": status,
                    "mission_type": self._get_mission_type(incident_type),
                    "fuel_level": np.random.randint(40, 100) if "fuel_capacity" in specs else None,
                    "battery_level": np.random.randint(40, 100) if "battery_capacity" in specs else None,
                    "crew_count": specs.get("crew_capacity", 0) if v_type not in ["Emergency_Drone", "Autonomous_Robot"] else 0,
                    "patient_onboard": status == "Transporting",
                    "patient_severity": incident_row.get("Incident_Severity") if status == "Transporting" else None,
                    "response_time_minutes": response_time,
                    "distance_to_incident": round(distance, 1),
                    "lights_sirens_active": status == "En_Route",
                    "dispatch_coordinator": incident_row.get("Dispatch_Coordinator", "AI"),
                    "linked_incident_id": incident_row.name if hasattr(incident_row, 'name') else None
                }
                
                # Add drone-specific fields
                if v_type == "Emergency_Drone":
                    record["payload_type"] = "AED" if incident_type == "Cardiac Arrest" else "Medical_Kit"
                    record["payload_weight"] = np.random.uniform(1, 5)
                    record["flight_mode"] = "Autonomous"
                
                records.append(record)
        
        return records
    
    def _get_mission_type(self, incident_type: str) -> str:
        mapping = {
            "Cardiac Arrest": "Cardiac_Arrest",
            "Accident": "Trauma",
            "Fire": "Fire_Response",
            "Other": "Medical_Emergency"
        }
        return mapping.get(incident_type, "Medical_Emergency")


class PVETDGenerator:
    """Private Vehicle Emergency Telemetry Dataset Generator"""
    
    def __init__(self, seed: int = Config.RANDOM_SEED):
        np.random.seed(seed)
    
    def generate_for_accident(self, incident_row: pd.Series) -> List[Dict]:
        """Generate PVETD records for accident incidents"""
        if incident_row.get("Incident_Type") != "Accident":
            return []
        
        records = []
        region = incident_row.get("Region_Type", "Urban")
        center = Config.REGION_CENTERS.get(region, Config.REGION_CENTERS["Urban"])
        
        severity_mapping = {
            "Low": "Minor",
            "Medium": "Moderate", 
            "High": "Severe"
        }
        crash_severity = severity_mapping.get(incident_row.get("Incident_Severity", "Low"), "Minor")
        
        # Generate 1-3 vehicles involved
        num_vehicles = np.random.randint(1, 4)
        
        for i in range(num_vehicles):
            vehicle_id = f"PV_{np.random.randint(10000, 99999)}"
            
            # Crash dynamics
            delta_v = {
                "Minor": np.random.uniform(8, 15),
                "Moderate": np.random.uniform(16, 30),
                "Severe": np.random.uniform(31, 50)
            }[crash_severity]
            
            record = {
                "record_id": f"PVETD_{np.random.randint(100000, 999999)}",
                "vehicle_id": vehicle_id,
                "vehicle_category": np.random.choice(["Sedan", "SUV", "Hatchback", "Truck"]),
                "timestamp": incident_row.get("Timestamp", datetime.now()),
                "latitude": round(center["lat"] + np.random.normal(0, 0.005), 6),
                "longitude": round(center["lon"] + np.random.normal(0, 0.005), 6),
                "region_type": region,
                "road_type": incident_row.get("Road_Type", "Local Road"),
                "speed_at_impact": np.random.randint(20, 100),
                "crash_detected": True,
                "crash_severity": crash_severity,
                "crash_delta_v": round(delta_v, 1),
                "crash_direction": np.random.choice(["Front", "Rear", "Left", "Right"]),
                "airbags_deployed": crash_severity in ["Moderate", "Severe"],
                "seatbelt_fastened": np.random.random() > 0.1,
                "occupant_count": np.random.randint(1, 5),
                "ecall_transmitted": True,
                "emergency_trigger": "Automatic_Crash",
                "estimated_injury_probability": round(np.random.uniform(0.3, 0.9) if crash_severity != "Minor" else np.random.uniform(0.05, 0.3), 2),
                "vehicle_operational": crash_severity == "Minor",
                "fire_detected": crash_severity == "Severe" and np.random.random() < 0.1,
                "hazard_lights_activated": True,
                "linked_incident_id": incident_row.name if hasattr(incident_row, 'name') else None
            }
            records.append(record)
        
        return records


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class IERDAGenerator:
    """Main orchestrator for generating all synthetic datasets"""
    
    def __init__(self, ierad_path: str = None, seed: int = Config.RANDOM_SEED):
        np.random.seed(seed)
        self.ierad = self._load_ierad(ierad_path)
        self.whmd_gen = WHMDGenerator(seed)
        self.ftd_gen = FTDGenerator(seed)
        self.pvetd_gen = PVETDGenerator(seed)
        
        print(f"✓ Initialized IERDA Generator")
        print(f"  - IERAD records: {len(self.ierad):,}")
        print(f"  - User population: {len(self.whmd_gen.users):,}")
        print(f"  - Fleet size: {len(self.ftd_gen.fleet):,}")
    
    def _load_ierad(self, path: str = None) -> pd.DataFrame:
        """Load IERAD dataset"""
        if path is None:
            # Try to download from Kaggle
            try:
                import kagglehub
                print("Downloading IERAD from Kaggle...")
                path = kagglehub.dataset_download("datasetengineer/integrated-emergency-response-dataset-ierad")
                # Find CSV file
                for f in os.listdir(path):
                    if f.endswith('.csv'):
                        path = os.path.join(path, f)
                        break
            except Exception as e:
                print(f"Could not download from Kaggle: {e}")
                print("Please provide IERAD path manually.")
                raise
        
        print(f"Loading IERAD from: {path}")
        df = pd.read_csv(path)
        
        # Parse timestamp if exists
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        return df
    
    def generate_all(self, sample_size: int = None, output_dir: str = "./output") -> Dict[str, pd.DataFrame]:
        """Generate all synthetic datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample IERAD if needed
        ierad_sample = self.ierad if sample_size is None else self.ierad.sample(n=min(sample_size, len(self.ierad)))
        
        print(f"\nGenerating datasets for {len(ierad_sample):,} IERAD records...")
        
        whmd_records = []
        ftd_records = []
        pvetd_records = []
        
        # Process each incident
        for idx, (_, incident) in enumerate(ierad_sample.iterrows()):
            if idx % 1000 == 0:
                print(f"  Processing incident {idx:,}/{len(ierad_sample):,}...")
            
            # Generate WHMD records
            num_affected = incident.get("Number_of_Injuries", 1)
            whmd_records.extend(self.whmd_gen.generate_for_incident(incident, num_affected))
            
            # Generate FTD records
            ftd_records.extend(self.ftd_gen.generate_for_incident(incident))
            
            # Generate PVETD records (only for accidents)
            pvetd_records.extend(self.pvetd_gen.generate_for_accident(incident))
        
        # Add some routine WHMD records
        print("  Adding routine monitoring records...")
        unique_timestamps = ierad_sample['Timestamp'].dropna().unique()[:100]
        for ts in unique_timestamps:
            whmd_records.extend(self.whmd_gen.generate_routine_records(ts, sample_size=50))
        
        # Convert to DataFrames
        whmd_df = pd.DataFrame(whmd_records)
        ftd_df = pd.DataFrame(ftd_records)
        pvetd_df = pd.DataFrame(pvetd_records)
        
        # Save to files
        print(f"\nSaving datasets to {output_dir}/")
        
        whmd_df.to_csv(os.path.join(output_dir, "WHMD_synthetic.csv"), index=False)
        ftd_df.to_csv(os.path.join(output_dir, "FTD_synthetic.csv"), index=False)
        pvetd_df.to_csv(os.path.join(output_dir, "PVETD_synthetic.csv"), index=False)
        
        # Save summary statistics
        summary = {
            "generation_timestamp": datetime.now().isoformat(),
            "ierad_records_processed": len(ierad_sample),
            "whmd_records": len(whmd_df),
            "ftd_records": len(ftd_df),
            "pvetd_records": len(pvetd_df),
            "user_population": len(self.whmd_gen.users),
            "fleet_size": len(self.ftd_gen.fleet)
        }
        
        pd.DataFrame([summary]).to_csv(os.path.join(output_dir, "generation_summary.csv"), index=False)
        
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"  WHMD records:  {len(whmd_df):,}")
        print(f"  FTD records:   {len(ftd_df):,}")
        print(f"  PVETD records: {len(pvetd_df):,}")
        print(f"\nFiles saved to: {output_dir}/")
        
        return {
            "WHMD": whmd_df,
            "FTD": ftd_df,
            "PVETD": pvetd_df,
            "IERAD_sample": ierad_sample
        }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate IERDA synthetic datasets")
    parser.add_argument("--ierad_path", type=str, default=None, 
                       help="Path to IERAD CSV file (will download from Kaggle if not provided)")
    parser.add_argument("--output_dir", type=str, default="./ierda_output",
                       help="Output directory for generated datasets")
    parser.add_argument("--sample_size", type=int, default=5000,
                       help="Number of IERAD records to process (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("="*60)
    print("IERDA - Integrated Emergency Response Data Architecture")
    print("Synthetic Dataset Generator v1.0")
    print("="*60)
    
    generator = IERDAGenerator(ierad_path=args.ierad_path, seed=args.seed)
    datasets = generator.generate_all(sample_size=args.sample_size, output_dir=args.output_dir)
    
    print("\n✓ All datasets generated successfully!")
