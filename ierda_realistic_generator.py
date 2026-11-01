"""
IERDA - Realistic Synthetic Data Generator
Generates more realistic synthetic data with proper noise and uncertainty
to avoid overfitting and produce academically credible ML results.

Authors: M. Kutlu Sengul, Vahap Tecim
Dokuz Eylul University, Management Information Systems
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class RealisticConfig:
    """Configuration for realistic data generation"""
    RANDOM_SEED = 42
    
    # Noise levels for realistic data
    NOISE_LEVEL = {
        'vital_signs': 0.15,      # 15% noise in vital measurements
        'sensor_error': 0.10,     # 10% sensor measurement error
        'label_noise': 0.08,      # 8% label noise (misclassification in real data)
        'missing_data': 0.05,     # 5% missing values
        'temporal_drift': 0.05    # 5% temporal drift in measurements
    }
    
    # User population
    TOTAL_USERS = 2000
    USER_TYPE_DISTRIBUTION = {
        "Normal": 0.50,
        "Chronic_Patient": 0.15,
        "Elderly": 0.15,
        "Athlete": 0.10,
        "Swimmer": 0.05,
        "High_Risk_Worker": 0.05
    }
    
    # Fleet composition
    FLEET_SIZE = 150
    FLEET_COMPOSITION = {
        "Ambulance": 0.25,
        "Emergency_Motorcycle": 0.12,
        "Fire_Truck": 0.15,
        "Rescue_Vehicle": 0.08,
        "Emergency_Drone": 0.20,
        "Autonomous_Robot": 0.05,
        "Emergency_Bicycle": 0.15
    }
    
    # Geographic bounds (Sydney)
    SYDNEY_BOUNDS = {
        "lat_min": -34.2, "lat_max": -33.5,
        "lon_min": 150.5, "lon_max": 151.5
    }

# =============================================================================
# REALISTIC USER PROFILES
# =============================================================================

REALISTIC_USER_PROFILES = {
    "Normal": {
        "age_range": (18, 65),
        "baseline_hr": (58, 82),
        "hr_variability": 12,  # Natural HR variability
        "baseline_spo2": (95, 100),
        "spo2_variability": 2,
        "baseline_temp": (36.1, 37.2),
        "temp_variability": 0.4,
        "activity_weights": {"Stationary": 0.45, "Walking": 0.30, "Running": 0.10, "Sleeping": 0.15},
        "emergency_base_prob": 0.002,
        "false_positive_rate": 0.05,  # Normal readings flagged as emergency
        "false_negative_rate": 0.10,  # Emergency missed
    },
    "Chronic_Patient": {
        "age_range": (40, 85),
        "baseline_hr": (62, 98),
        "hr_variability": 18,
        "baseline_spo2": (88, 97),
        "spo2_variability": 4,
        "baseline_temp": (35.8, 37.5),
        "temp_variability": 0.6,
        "activity_weights": {"Stationary": 0.60, "Walking": 0.25, "Running": 0.02, "Sleeping": 0.13},
        "emergency_base_prob": 0.025,
        "false_positive_rate": 0.12,
        "false_negative_rate": 0.08,
    },
    "Elderly": {
        "age_range": (65, 95),
        "baseline_hr": (52, 88),
        "hr_variability": 15,
        "baseline_spo2": (90, 98),
        "spo2_variability": 3,
        "baseline_temp": (35.5, 37.0),
        "temp_variability": 0.5,
        "activity_weights": {"Stationary": 0.65, "Walking": 0.20, "Running": 0.00, "Sleeping": 0.15},
        "emergency_base_prob": 0.020,
        "fall_risk_multiplier": 4.0,
        "false_positive_rate": 0.15,
        "false_negative_rate": 0.12,
    },
    "Athlete": {
        "age_range": (16, 45),
        "baseline_hr": (42, 68),
        "hr_variability": 20,
        "baseline_spo2": (96, 100),
        "spo2_variability": 2,
        "baseline_temp": (35.8, 38.0),
        "temp_variability": 0.8,
        "activity_weights": {"Stationary": 0.25, "Walking": 0.15, "Running": 0.35, "Sleeping": 0.25},
        "emergency_base_prob": 0.005,
        "false_positive_rate": 0.08,
        "false_negative_rate": 0.05,
    },
    "Swimmer": {
        "age_range": (10, 70),
        "baseline_hr": (52, 78),
        "hr_variability": 15,
        "baseline_spo2": (94, 100),
        "spo2_variability": 3,
        "baseline_temp": (35.5, 37.5),
        "temp_variability": 0.7,
        "activity_weights": {"Stationary": 0.35, "Walking": 0.20, "Swimming": 0.30, "Sleeping": 0.15},
        "emergency_base_prob": 0.008,
        "drowning_risk": 0.003,
        "false_positive_rate": 0.10,
        "false_negative_rate": 0.15,
    },
    "High_Risk_Worker": {
        "age_range": (18, 65),
        "baseline_hr": (58, 88),
        "hr_variability": 18,
        "baseline_spo2": (92, 99),
        "spo2_variability": 4,
        "baseline_temp": (36.0, 38.0),
        "temp_variability": 0.8,
        "activity_weights": {"Stationary": 0.15, "Walking": 0.45, "Running": 0.10, "Working": 0.30},
        "emergency_base_prob": 0.012,
        "hazard_exposure_prob": 0.02,
        "false_positive_rate": 0.10,
        "false_negative_rate": 0.08,
    }
}

# =============================================================================
# REALISTIC VEHICLE SPECIFICATIONS
# =============================================================================

REALISTIC_VEHICLE_SPECS = {
    "Ambulance": {
        "speed_range": (15, 95),
        "avg_speed_urban": 35,
        "avg_speed_rural": 70,
        "speed_variability": 15,
        "response_time_urban": (5, 15),  # Range instead of fixed
        "response_time_rural": (10, 30),
        "fuel_consumption_range": (12, 20),
        "maintenance_prob_base": 0.02,
        "breakdown_prob": 0.005,
    },
    "Emergency_Motorcycle": {
        "speed_range": (20, 130),
        "avg_speed_urban": 50,
        "avg_speed_rural": 90,
        "speed_variability": 25,
        "response_time_urban": (3, 10),
        "response_time_rural": (8, 18),
        "fuel_consumption_range": (4, 8),
        "maintenance_prob_base": 0.03,
        "breakdown_prob": 0.008,
    },
    "Fire_Truck": {
        "speed_range": (10, 85),
        "avg_speed_urban": 30,
        "avg_speed_rural": 60,
        "speed_variability": 12,
        "response_time_urban": (5, 12),
        "response_time_rural": (10, 25),
        "fuel_consumption_range": (25, 45),
        "maintenance_prob_base": 0.025,
        "breakdown_prob": 0.004,
    },
    "Rescue_Vehicle": {
        "speed_range": (15, 100),
        "avg_speed_urban": 40,
        "avg_speed_rural": 75,
        "speed_variability": 18,
        "response_time_urban": (6, 15),
        "response_time_rural": (12, 28),
        "fuel_consumption_range": (15, 28),
        "maintenance_prob_base": 0.02,
        "breakdown_prob": 0.006,
    },
    "Emergency_Drone": {
        "speed_range": (25, 110),
        "avg_speed_urban": 70,
        "avg_speed_rural": 85,
        "speed_variability": 20,
        "response_time_urban": (2, 8),
        "response_time_rural": (4, 12),
        "battery_drain_rate": (1.5, 3.5),
        "maintenance_prob_base": 0.04,
        "breakdown_prob": 0.015,
    },
    "Autonomous_Robot": {
        "speed_range": (3, 28),
        "avg_speed_urban": 15,
        "avg_speed_rural": 10,
        "speed_variability": 5,
        "response_time_urban": (10, 25),
        "response_time_rural": (20, 45),
        "battery_drain_rate": (0.8, 2.0),
        "maintenance_prob_base": 0.05,
        "breakdown_prob": 0.02,
    },
    "Emergency_Bicycle": {
        "speed_range": (8, 45),
        "avg_speed_urban": 25,
        "avg_speed_rural": 20,
        "speed_variability": 10,
        "response_time_urban": (3, 10),
        "response_time_rural": (8, 20),
        "battery_drain_rate": (0.5, 1.5),
        "maintenance_prob_base": 0.02,
        "breakdown_prob": 0.01,
    }
}

# =============================================================================
# REALISTIC DATA GENERATORS
# =============================================================================

class RealisticWHMDGenerator:
    """Realistic Wearable Health Monitoring Dataset Generator"""
    
    def __init__(self, seed: int = RealisticConfig.RANDOM_SEED):
        np.random.seed(seed)
        self.users = self._create_user_base()
        
    def _create_user_base(self) -> pd.DataFrame:
        """Create base user population with individual characteristics"""
        users = []
        user_id = 0
        
        for user_type, proportion in RealisticConfig.USER_TYPE_DISTRIBUTION.items():
            count = int(RealisticConfig.TOTAL_USERS * proportion)
            profile = REALISTIC_USER_PROFILES[user_type]
            
            for _ in range(count):
                user_id += 1
                age = np.random.randint(profile["age_range"][0], profile["age_range"][1])
                
                # Individual baseline (person-specific)
                personal_hr_offset = np.random.normal(0, 5)
                personal_spo2_offset = np.random.normal(0, 1)
                
                users.append({
                    "user_id": f"USR_{user_id:05d}",
                    "user_type": user_type,
                    "age": age,
                    "personal_hr_offset": personal_hr_offset,
                    "personal_spo2_offset": personal_spo2_offset,
                    "baseline_hr_low": profile["baseline_hr"][0],
                    "baseline_hr_high": profile["baseline_hr"][1],
                    "hr_variability": profile["hr_variability"],
                    "baseline_spo2_low": profile["baseline_spo2"][0],
                    "baseline_spo2_high": profile["baseline_spo2"][1],
                    "spo2_variability": profile["spo2_variability"],
                    "emergency_base_prob": profile["emergency_base_prob"],
                    "false_positive_rate": profile["false_positive_rate"],
                    "false_negative_rate": profile["false_negative_rate"],
                    "fall_risk": profile.get("fall_risk_multiplier", 1.0),
                })
        
        return pd.DataFrame(users)
    
    def _add_measurement_noise(self, value: float, noise_level: float) -> float:
        """Add realistic measurement noise"""
        noise = np.random.normal(0, noise_level * abs(value))
        return value + noise
    
    def _simulate_sensor_error(self, value: float, error_prob: float = 0.03) -> Tuple[float, bool]:
        """Simulate occasional sensor errors"""
        if np.random.random() < error_prob:
            # Sensor malfunction - extreme or missing value
            if np.random.random() < 0.5:
                return value * np.random.uniform(0.5, 1.5), True
            else:
                return np.nan, True
        return value, False
    
    def generate_realistic_record(self, user: pd.Series, timestamp: datetime, 
                                  incident_context: dict = None) -> dict:
        """Generate a single realistic health record"""
        profile = REALISTIC_USER_PROFILES[user["user_type"]]
        
        # Determine if this is a true emergency (ground truth) - FIRST
        # This is determined by latent/hidden factors, not directly by observables
        is_true_emergency = False
        emergency_type = None
        
        # Hidden risk factors (not directly observable)
        hidden_health_risk = np.random.beta(2, 10)  # Most people are healthy
        age_risk_factor = (user["age"] - 18) / 80  # Older = higher risk
        user_type_risk = user["emergency_base_prob"] * 10
        
        # True emergency probability based on hidden factors
        true_emergency_prob = (
            hidden_health_risk * 0.5 +
            age_risk_factor * 0.2 +
            user_type_risk * 0.3 +
            np.random.uniform(0, 0.1)  # Random component
        )
        true_emergency_prob = np.clip(true_emergency_prob, 0, 1)
        
        if incident_context:
            true_emergency_prob += 0.3  # Incident increases probability
        
        is_true_emergency = np.random.random() < true_emergency_prob * 0.15
        
        # Generate vital signs - INDEPENDENT of emergency status (with some correlation)
        # These are noisy observations of underlying health
        
        # Base vitals from user profile
        hr_base = np.random.randint(user["baseline_hr_low"], user["baseline_hr_high"])
        hr_base += user["personal_hr_offset"]
        spo2_base = np.random.randint(user["baseline_spo2_low"], user["baseline_spo2_high"])
        spo2_base += user["personal_spo2_offset"]
        temp_base = np.random.uniform(36.0, 37.2)
        stress_base = np.random.randint(5, 45)
        resp_rate = np.random.randint(12, 20)
        
        # If emergency, vitals MAY be affected (but not always - that's the challenge)
        # Only ~60-70% of emergencies show clear vital sign changes
        if is_true_emergency and np.random.random() < 0.65:
            # Vital signs are affected
            hr_modifier = np.random.uniform(20, 80)
            spo2_modifier = np.random.uniform(-15, -3)
            temp_modifier = np.random.uniform(-1.5, 2.5)
            stress_modifier = np.random.uniform(30, 60)
            resp_modifier = np.random.uniform(5, 15)
        else:
            # Normal variation (even in some emergencies)
            hr_modifier = np.random.normal(0, 10)
            spo2_modifier = np.random.normal(0, 2)
            temp_modifier = np.random.normal(0, 0.3)
            stress_modifier = np.random.normal(0, 10)
            resp_modifier = np.random.normal(0, 3)
        
        # Apply modifiers
        hr = hr_base + hr_modifier
        spo2 = spo2_base + spo2_modifier
        temp = temp_base + temp_modifier
        stress = stress_base + stress_modifier
        resp_rate = resp_rate + resp_modifier
        
        # Add measurement noise
        hr = self._add_measurement_noise(hr, RealisticConfig.NOISE_LEVEL['vital_signs'])
        spo2 = self._add_measurement_noise(spo2, RealisticConfig.NOISE_LEVEL['vital_signs'])
        temp = self._add_measurement_noise(temp, RealisticConfig.NOISE_LEVEL['vital_signs'] * 0.5)
        
        # Simulate sensor errors
        hr, hr_error = self._simulate_sensor_error(hr)
        spo2, spo2_error = self._simulate_sensor_error(spo2)
        
        # Clamp values
        hr = np.clip(hr, 30, 220) if not np.isnan(hr) else np.nan
        spo2 = np.clip(spo2, 70, 100) if not np.isnan(spo2) else np.nan
        temp = np.clip(temp, 34.0, 42.0)
        stress = np.clip(stress, 0, 100)
        resp_rate = np.clip(resp_rate, 8, 40)
        
        # Determine detected risk level (what the system thinks)
        # Based on observable vitals - NOT ground truth
        vital_anomaly_score = 0
        if not np.isnan(hr):
            if hr > 120 or hr < 50:
                vital_anomaly_score += 2
            elif hr > 100 or hr < 55:
                vital_anomaly_score += 1
        
        if not np.isnan(spo2):
            if spo2 < 90:
                vital_anomaly_score += 3
            elif spo2 < 94:
                vital_anomaly_score += 1
        
        if temp > 38.5 or temp < 35.5:
            vital_anomaly_score += 2
        elif temp > 37.8 or temp < 36:
            vital_anomaly_score += 1
        
        if stress > 75:
            vital_anomaly_score += 1
        
        # Add noise to detection
        vital_anomaly_score += np.random.normal(0, 1)
        
        if vital_anomaly_score >= 5:
            detected_risk = "Critical"
        elif vital_anomaly_score >= 3:
            detected_risk = "High"
        elif vital_anomaly_score >= 1.5:
            detected_risk = "Elevated"
        else:
            detected_risk = "Normal"
        
        # Emergency probability (model's noisy estimate)
        emergency_prob = vital_anomaly_score / 8 + np.random.normal(0, 0.15)
        emergency_prob = np.clip(emergency_prob, 0.01, 0.99)
        
        # Blood pressure
        bp_sys = np.random.randint(100, 150) + (15 if is_true_emergency and np.random.random() < 0.5 else 0)
        bp_dia = np.random.randint(60, 95) + (10 if is_true_emergency and np.random.random() < 0.5 else 0)
        
        # Activity
        activity = np.random.choice(
            list(profile["activity_weights"].keys()),
            p=list(profile["activity_weights"].values())
        )
        
        # Fall detection (noisy)
        fall_detected = False
        if user["user_type"] == "Elderly":
            fall_detected = np.random.random() < 0.002
        if is_true_emergency and np.random.random() < 0.15:
            fall_detected = True
        
        # SOS (noisy - some people press even without emergency)
        sos_pressed = np.random.random() < 0.005
        if is_true_emergency:
            sos_pressed = np.random.random() < 0.25
        
        # ECG
        ecg_abnormal = is_true_emergency and np.random.random() < 0.20
        ecg_status = np.random.choice(["AFib_Detected", "Abnormal"], p=[0.6, 0.4]) if ecg_abnormal else "Normal"
        if np.random.random() < 0.08:
            ecg_status = "Inconclusive"
        
        # Region
        region = np.random.choice(["Urban", "Suburban", "Rural"], p=[0.55, 0.30, 0.15])
        center = {"Urban": (-33.87, 151.21), "Suburban": (-33.82, 151.00), "Rural": (-33.70, 150.70)}[region]
        
        record = {
            "record_id": f"WHMD_{np.random.randint(100000, 999999)}",
            "user_id": user["user_id"],
            "user_type": user["user_type"],
            "age": user["age"],
            "timestamp": timestamp,
            "latitude": round(center[0] + np.random.normal(0, 0.03), 6),
            "longitude": round(center[1] + np.random.normal(0, 0.03), 6),
            "region_type": region,
            
            # Vital signs
            "heart_rate": round(hr, 0) if not np.isnan(hr) else np.nan,
            "heart_rate_variability": round(np.random.uniform(15, 85), 1),
            "blood_oxygen": round(spo2, 1) if not np.isnan(spo2) else np.nan,
            "body_temperature": round(temp, 1),
            "respiratory_rate": int(resp_rate),
            "stress_level": int(stress),
            "blood_pressure_systolic": bp_sys,
            "blood_pressure_diastolic": bp_dia,
            
            # Activity
            "activity_type": activity,
            "steps_count": np.random.randint(0, 800) if activity != "Sleeping" else 0,
            "inactivity_duration": np.random.randint(0, 60) if activity == "Stationary" else 0,
            
            # Alerts
            "ecg_status": ecg_status,
            "fall_detected": fall_detected,
            "sos_button_pressed": sos_pressed,
            
            # Device
            "device_battery": np.random.randint(15, 100),
            "connection_status": np.random.choice(["Connected", "Connected", "Intermittent", "Disconnected"], 
                                                  p=[0.80, 0.10, 0.08, 0.02]),
            "sensor_error": hr_error or spo2_error,
            
            # System's prediction (noisy)
            "risk_level": detected_risk,
            "emergency_probability": round(emergency_prob, 4),
            
            # Ground truth (for ML training/validation)
            "true_emergency": is_true_emergency,
            
            "linked_incident_id": incident_context.get("incident_id") if incident_context else None
        }
        
        return record
    
    def generate_dataset(self, n_records: int, ierad_df: pd.DataFrame = None) -> pd.DataFrame:
        """Generate complete WHMD dataset"""
        records = []
        
        # Generate base timestamps
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 12, 31)
        total_days = (end_date - start_date).days
        
        print(f"Generating {n_records:,} WHMD records...")
        
        for i in range(n_records):
            if i % 5000 == 0:
                print(f"  Progress: {i:,}/{n_records:,}")
            
            # Select random user
            user = self.users.sample(n=1).iloc[0]
            
            # Generate random timestamp
            random_day = int(np.random.randint(0, total_days))
            random_hour = int(np.random.randint(0, 24))
            random_minute = int(np.random.choice([0, 10, 20, 30, 40, 50]))
            timestamp = start_date + timedelta(days=random_day, hours=random_hour, minutes=random_minute)
            
            # Occasionally link to IERAD incident
            incident_context = None
            if ierad_df is not None and np.random.random() < 0.1:
                incident = ierad_df.sample(n=1).iloc[0]
                incident_context = {
                    "incident_id": incident.name,
                    "type": incident.get("Incident_Type", "general")
                }
            
            record = self.generate_realistic_record(user, timestamp, incident_context)
            records.append(record)
        
        return pd.DataFrame(records)


class RealisticFTDGenerator:
    """Realistic Fleet Telemetry Dataset Generator"""
    
    def __init__(self, seed: int = RealisticConfig.RANDOM_SEED):
        np.random.seed(seed)
        self.fleet = self._create_fleet()
        
    def _create_fleet(self) -> pd.DataFrame:
        """Create vehicle fleet with individual characteristics"""
        vehicles = []
        vehicle_id = 0
        
        for v_type, proportion in RealisticConfig.FLEET_COMPOSITION.items():
            count = int(RealisticConfig.FLEET_SIZE * proportion)
            specs = REALISTIC_VEHICLE_SPECS[v_type]
            
            for i in range(count):
                vehicle_id += 1
                
                # Vehicle age affects maintenance probability
                vehicle_age = np.random.randint(0, 10)
                maintenance_multiplier = 1 + (vehicle_age * 0.15)
                
                vehicles.append({
                    "vehicle_id": f"VEH_{vehicle_id:04d}",
                    "vehicle_type": v_type,
                    "callsign": f"{v_type[:3].upper()}-{i+1:02d}",
                    "vehicle_age_years": vehicle_age,
                    "maintenance_multiplier": maintenance_multiplier,
                    "avg_speed_urban": specs["avg_speed_urban"],
                    "avg_speed_rural": specs["avg_speed_rural"],
                    "speed_variability": specs["speed_variability"],
                    "maintenance_prob": specs["maintenance_prob_base"] * maintenance_multiplier,
                    "breakdown_prob": specs["breakdown_prob"] * maintenance_multiplier,
                })
        
        return pd.DataFrame(vehicles)
    
    def generate_realistic_record(self, vehicle: pd.Series, timestamp: datetime,
                                  mission_context: dict = None) -> dict:
        """Generate a single realistic fleet telemetry record"""
        specs = REALISTIC_VEHICLE_SPECS[vehicle["vehicle_type"]]
        
        # Determine vehicle status
        if mission_context:
            status = np.random.choice(["En_Route", "On_Scene", "Transporting", "Returning"], 
                                     p=[0.30, 0.25, 0.25, 0.20])
        else:
            status = np.random.choice(["Available", "Maintenance", "En_Route", "On_Scene"],
                                     p=[0.50, 0.10, 0.25, 0.15])
        
        # Region affects speed
        region = np.random.choice(["Urban", "Suburban", "Rural"], p=[0.50, 0.30, 0.20])
        
        if region == "Urban":
            base_speed = vehicle["avg_speed_urban"]
        else:
            base_speed = vehicle["avg_speed_rural"]
        
        # Add realistic speed variation
        speed = base_speed + np.random.normal(0, vehicle["speed_variability"])
        speed = np.clip(speed, specs["speed_range"][0], specs["speed_range"][1])
        
        # Response time with realistic variation
        if status in ["En_Route", "On_Scene"]:
            rt_range = specs["response_time_urban"] if region == "Urban" else specs["response_time_rural"]
            response_time = np.random.uniform(rt_range[0], rt_range[1])
            # Add traffic/condition effects
            response_time *= np.random.uniform(0.8, 1.4)
        else:
            response_time = None
        
        # Fuel/Battery level with degradation
        is_electric = vehicle["vehicle_type"] in ["Emergency_Drone", "Autonomous_Robot", "Emergency_Bicycle"]
        
        if is_electric:
            battery_level = np.random.uniform(20, 100)
            # Degradation based on activity
            if status in ["En_Route", "On_Scene", "Transporting"]:
                battery_level -= np.random.uniform(5, 20)
            fuel_level = None
        else:
            fuel_level = np.random.uniform(25, 100)
            if status in ["En_Route", "Transporting"]:
                fuel_level -= np.random.uniform(5, 15)
            battery_level = None
        
        # Maintenance prediction - key for ML
        # This should NOT be perfectly predictable
        
        # Base probability from vehicle characteristics
        maint_prob = vehicle["maintenance_prob"]
        
        # Factors that increase maintenance need (with noise)
        age_factor = vehicle["vehicle_age_years"] * 0.02
        usage_factor = 0.05 if status in ["En_Route", "Transporting"] else 0
        fuel_factor = 0.1 if (fuel_level and fuel_level < 30) else 0
        battery_factor = 0.15 if (battery_level and battery_level < 25) else 0
        
        # Random factors (unknown in real scenarios)
        random_factor = np.random.uniform(-0.05, 0.15)
        
        maintenance_probability = maint_prob + age_factor + usage_factor + fuel_factor + battery_factor + random_factor
        maintenance_probability = np.clip(maintenance_probability, 0, 1)
        
        # Actual maintenance need (ground truth) - probabilistic
        needs_maintenance = np.random.random() < maintenance_probability
        
        # Detected maintenance status (what sensors/system detect - with error)
        if needs_maintenance:
            # True positive rate ~80%
            detected_maintenance = np.random.random() < 0.80
        else:
            # False positive rate ~10%
            detected_maintenance = np.random.random() < 0.10
        
        # Breakdown event (rare but serious)
        breakdown = np.random.random() < vehicle["breakdown_prob"]
        
        # Distance to incident
        distance = np.random.uniform(0.5, 25) if status in ["En_Route"] else np.random.uniform(0, 5)
        
        # Location
        center = {"Urban": (-33.87, 151.21), "Suburban": (-33.82, 151.00), "Rural": (-33.70, 150.70)}[region]
        
        record = {
            "record_id": f"FTD_{np.random.randint(100000, 999999)}",
            "vehicle_id": vehicle["vehicle_id"],
            "vehicle_type": vehicle["vehicle_type"],
            "callsign": vehicle["callsign"],
            "vehicle_age_years": vehicle["vehicle_age_years"],
            "timestamp": timestamp,
            
            # Location
            "latitude": round(center[0] + np.random.normal(0, 0.02), 6),
            "longitude": round(center[1] + np.random.normal(0, 0.02), 6),
            "region_type": region,
            
            # Movement
            "speed": round(speed, 1),
            "heading": np.random.randint(0, 360),
            "altitude": np.random.randint(0, 150) if vehicle["vehicle_type"] == "Emergency_Drone" else 0,
            
            # Status
            "status": status,
            "mission_active": status in ["En_Route", "On_Scene", "Transporting"],
            
            # Energy
            "fuel_level": round(fuel_level, 1) if fuel_level else None,
            "battery_level": round(battery_level, 1) if battery_level else None,
            
            # Performance
            "response_time_minutes": round(response_time, 1) if response_time else None,
            "distance_to_incident": round(distance, 1),
            
            # Crew (for manned vehicles)
            "crew_count": np.random.randint(1, 4) if vehicle["vehicle_type"] not in ["Emergency_Drone", "Autonomous_Robot"] else 0,
            "patient_onboard": status == "Transporting" and np.random.random() < 0.8,
            
            # Vehicle health indicators
            "engine_temp": np.random.uniform(80, 110) if not is_electric else None,
            "oil_pressure": np.random.uniform(25, 65) if not is_electric else None,
            "tire_pressure_avg": np.random.uniform(30, 40) if vehicle["vehicle_type"] not in ["Emergency_Drone"] else None,
            
            # Maintenance - MODEL'S prediction (not ground truth)
            "maintenance_probability": round(maintenance_probability, 4),
            "detected_maintenance_needed": detected_maintenance,
            
            # Ground truth (for validation)
            "true_maintenance_needed": needs_maintenance,
            "breakdown_occurred": breakdown,
            
            # Operations
            "lights_sirens_active": status == "En_Route" and np.random.random() < 0.8,
            "dispatch_type": np.random.choice(["Human", "AI", "Hybrid"], p=[0.30, 0.50, 0.20]),
            
            "linked_incident_id": mission_context.get("incident_id") if mission_context else None
        }
        
        return record
    
    def generate_dataset(self, n_records: int, ierad_df: pd.DataFrame = None) -> pd.DataFrame:
        """Generate complete FTD dataset"""
        records = []
        
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 12, 31)
        total_days = (end_date - start_date).days
        
        print(f"Generating {n_records:,} FTD records...")
        
        for i in range(n_records):
            if i % 5000 == 0:
                print(f"  Progress: {i:,}/{n_records:,}")
            
            vehicle = self.fleet.sample(n=1).iloc[0]
            
            random_day = np.random.randint(0, total_days)
            random_hour = np.random.randint(0, 24)
            timestamp = start_date + timedelta(days=random_day, hours=random_hour)
            
            mission_context = None
            if ierad_df is not None and np.random.random() < 0.3:
                incident = ierad_df.sample(n=1).iloc[0]
                mission_context = {"incident_id": incident.name}
            
            record = self.generate_realistic_record(vehicle, timestamp, mission_context)
            records.append(record)
        
        return pd.DataFrame(records)


class RealisticPVETDGenerator:
    """Realistic Private Vehicle Emergency Telemetry Dataset Generator"""
    
    def __init__(self, seed: int = RealisticConfig.RANDOM_SEED):
        np.random.seed(seed)
    
    def generate_realistic_accident(self, timestamp: datetime, region: str) -> dict:
        """Generate a realistic accident record"""
        
        # Vehicle characteristics
        vehicle_category = np.random.choice(
            ["Sedan", "SUV", "Hatchback", "Truck", "Motorcycle"],
            p=[0.35, 0.30, 0.15, 0.12, 0.08]
        )
        vehicle_year = np.random.randint(2005, 2025)
        vehicle_age = 2024 - vehicle_year
        
        # Speed at impact - varies by road type and conditions
        road_type = np.random.choice(["Highway", "Local Road", "Unpaved Road"], p=[0.30, 0.60, 0.10])
        
        if road_type == "Highway":
            speed = np.random.uniform(60, 140)
        elif road_type == "Local Road":
            speed = np.random.uniform(20, 80)
        else:
            speed = np.random.uniform(10, 50)
        
        # Crash characteristics
        crash_direction = np.random.choice(["Front", "Rear", "Left", "Right", "Rollover"],
                                          p=[0.40, 0.25, 0.15, 0.15, 0.05])
        
        # Delta-V (change in velocity) - key crash severity indicator
        # NOT directly derived from speed - has its own distribution
        base_delta_v = np.random.exponential(15)  # Exponential distribution for crashes
        delta_v = np.clip(base_delta_v, 5, 80)
        
        # Add noise and variation
        delta_v *= np.random.uniform(0.7, 1.3)
        
        # TRUE severity based on multiple factors (ground truth)
        # This is the actual severity - not what sensors detect
        severity_score = (
            delta_v * 0.4 +                      # Delta-V is major factor
            speed * 0.15 +                        # Speed contributes
            (10 if crash_direction == "Rollover" else 0) +  # Rollovers are severe
            (5 if vehicle_category == "Motorcycle" else 0) +  # Motorcycles more vulnerable
            np.random.normal(0, 8)                # Random factors (unseen variables)
        )
        
        if severity_score < 25:
            true_severity = "Minor"
        elif severity_score < 50:
            true_severity = "Moderate"
        else:
            true_severity = "Severe"
        
        # DETECTED severity - what sensors/systems estimate (with error)
        # This is what the ML model would predict
        detection_noise = np.random.normal(0, 12)
        detected_score = severity_score + detection_noise
        
        if detected_score < 25:
            detected_severity = "Minor"
        elif detected_score < 50:
            detected_severity = "Moderate"
        else:
            detected_severity = "Severe"
        
        # Occupant and safety features
        occupant_count = np.random.choice([1, 2, 3, 4, 5], p=[0.40, 0.30, 0.15, 0.10, 0.05])
        seatbelt_rate = 0.92 if vehicle_year >= 2010 else 0.80
        seatbelt_fastened = np.random.random() < seatbelt_rate
        
        # Airbag deployment - depends on severity but not perfectly
        airbag_prob = {"Minor": 0.15, "Moderate": 0.60, "Severe": 0.95}[true_severity]
        airbags_deployed = np.random.random() < airbag_prob
        
        # Injury probability - complex relationship
        base_injury_prob = {
            "Minor": np.random.uniform(0.05, 0.25),
            "Moderate": np.random.uniform(0.25, 0.65),
            "Severe": np.random.uniform(0.55, 0.95)
        }[true_severity]
        
        # Modify by safety factors
        if seatbelt_fastened:
            base_injury_prob *= 0.6
        if airbags_deployed:
            base_injury_prob *= 0.75
        if vehicle_category == "Motorcycle":
            base_injury_prob *= 1.5
        
        injury_probability = np.clip(base_injury_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        
        # Vehicle post-crash status
        vehicle_operational = true_severity == "Minor" and np.random.random() < 0.7
        fire_detected = true_severity == "Severe" and np.random.random() < 0.08
        
        # Emergency response
        ecall_prob = 0.95 if vehicle_year >= 2018 else 0.60
        ecall_transmitted = np.random.random() < ecall_prob
        
        # Location
        center = {"Urban": (-33.87, 151.21), "Suburban": (-33.82, 151.00), "Rural": (-33.70, 150.70)}[region]
        
        record = {
            "record_id": f"PVETD_{np.random.randint(100000, 999999)}",
            "vehicle_id": f"PV_{np.random.randint(10000, 99999)}",
            "vehicle_category": vehicle_category,
            "vehicle_year": vehicle_year,
            "vehicle_age": vehicle_age,
            "timestamp": timestamp,
            
            # Location
            "latitude": round(center[0] + np.random.normal(0, 0.02), 6),
            "longitude": round(center[1] + np.random.normal(0, 0.02), 6),
            "region_type": region,
            "road_type": road_type,
            
            # Crash dynamics
            "speed_at_impact": round(speed, 1),
            "crash_delta_v": round(delta_v, 1),
            "crash_direction": crash_direction,
            
            # Detected values (model's view)
            "detected_severity": detected_severity,
            "estimated_injury_probability": round(injury_probability, 4),
            
            # Ground truth (for validation)
            "true_severity": true_severity,
            
            # Safety systems
            "occupant_count": occupant_count,
            "seatbelt_fastened": seatbelt_fastened,
            "airbags_deployed": airbags_deployed,
            
            # Vehicle status
            "vehicle_operational": vehicle_operational,
            "fire_detected": fire_detected,
            "fuel_leak_detected": true_severity == "Severe" and np.random.random() < 0.05,
            
            # Emergency systems
            "ecall_transmitted": ecall_transmitted,
            "manual_sos_pressed": np.random.random() < 0.15,
            "hazard_lights_activated": np.random.random() < 0.90,
            
            # Response recommendation (system's suggestion)
            "recommended_response": {
                "Minor": np.random.choice(["None", "Check", "Ambulance"], p=[0.20, 0.60, 0.20]),
                "Moderate": np.random.choice(["Check", "Ambulance", "Fire_Rescue"], p=[0.10, 0.70, 0.20]),
                "Severe": np.random.choice(["Ambulance", "Fire_Rescue"], p=[0.40, 0.60])
            }[detected_severity]
        }
        
        return record
    
    def generate_dataset(self, n_records: int) -> pd.DataFrame:
        """Generate complete PVETD dataset"""
        records = []
        
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 12, 31)
        total_days = (end_date - start_date).days
        
        print(f"Generating {n_records:,} PVETD records...")
        
        for i in range(n_records):
            if i % 2000 == 0:
                print(f"  Progress: {i:,}/{n_records:,}")
            
            random_day = np.random.randint(0, total_days)
            random_hour = np.random.randint(0, 24)
            timestamp = start_date + timedelta(days=random_day, hours=random_hour)
            
            region = np.random.choice(["Urban", "Suburban", "Rural"], p=[0.50, 0.35, 0.15])
            
            record = self.generate_realistic_accident(timestamp, region)
            records.append(record)
        
        return pd.DataFrame(records)


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_realistic_datasets(
    output_dir: str = "./realistic_output",
    whmd_records: int = 25000,
    ftd_records: int = 20000,
    pvetd_records: int = 10000,
    ierad_path: str = None,
    seed: int = 42
):
    """Generate all realistic datasets"""
    
    print("="*70)
    print("REALISTIC IERDA DATA GENERATOR")
    print("Generating academically credible synthetic datasets")
    print("="*70)
    
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load IERAD for linking
    ierad_df = None
    if ierad_path:
        print(f"\nLoading IERAD from: {ierad_path}")
        ierad_df = pd.read_csv(ierad_path)
        print(f"  Loaded {len(ierad_df):,} IERAD records")
    
    # Generate WHMD
    print("\n" + "-"*50)
    whmd_gen = RealisticWHMDGenerator(seed)
    whmd_df = whmd_gen.generate_dataset(whmd_records, ierad_df)
    whmd_df.to_csv(os.path.join(output_dir, "WHMD_realistic.csv"), index=False)
    print(f"✓ WHMD: {len(whmd_df):,} records saved")
    
    # Generate FTD
    print("\n" + "-"*50)
    ftd_gen = RealisticFTDGenerator(seed)
    ftd_df = ftd_gen.generate_dataset(ftd_records, ierad_df)
    ftd_df.to_csv(os.path.join(output_dir, "FTD_realistic.csv"), index=False)
    print(f"✓ FTD: {len(ftd_df):,} records saved")
    
    # Generate PVETD
    print("\n" + "-"*50)
    pvetd_gen = RealisticPVETDGenerator(seed)
    pvetd_df = pvetd_gen.generate_dataset(pvetd_records)
    pvetd_df.to_csv(os.path.join(output_dir, "PVETD_realistic.csv"), index=False)
    print(f"✓ PVETD: {len(pvetd_df):,} records saved")
    
    # Summary statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print("\nWHMD Risk Distribution:")
    print(whmd_df['risk_level'].value_counts(normalize=True).round(3))
    print(f"True emergency rate: {whmd_df['true_emergency'].mean():.3f}")
    
    print("\nFTD Maintenance Distribution:")
    print(f"True maintenance needed: {ftd_df['true_maintenance_needed'].mean():.3f}")
    print(f"Detected maintenance: {ftd_df['detected_maintenance_needed'].mean():.3f}")
    
    print("\nPVETD Severity Distribution:")
    print(pvetd_df['true_severity'].value_counts(normalize=True).round(3))
    print(f"Detection accuracy: {(pvetd_df['true_severity'] == pvetd_df['detected_severity']).mean():.3f}")
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print(f"All files saved to: {output_dir}/")
    print("="*70)
    
    return whmd_df, ftd_df, pvetd_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate realistic IERDA datasets")
    parser.add_argument("--output_dir", type=str, default="./realistic_output")
    parser.add_argument("--whmd_records", type=int, default=25000)
    parser.add_argument("--ftd_records", type=int, default=20000)
    parser.add_argument("--pvetd_records", type=int, default=10000)
    parser.add_argument("--ierad_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    generate_realistic_datasets(
        output_dir=args.output_dir,
        whmd_records=args.whmd_records,
        ftd_records=args.ftd_records,
        pvetd_records=args.pvetd_records,
        ierad_path=args.ierad_path,
        seed=args.seed
    )
