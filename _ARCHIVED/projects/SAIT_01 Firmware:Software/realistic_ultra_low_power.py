#!/usr/bin/env python3
"""
Realistic Ultra Low Power Design for 1-2 Year Battery Life
=========================================================

Redesigned power architecture for extended operation:
- Hardware-based wake-on-sound with analog comparator
- Extremely aggressive duty cycling
- Motion-activated enhanced monitoring
- Configurable sensitivity modes
- Primary lithium battery optimization

Realistic power targets without sacrificing core performance.
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealisticPowerManager:
    """Realistic ultra-low power management for 1-2 year operation"""
    
    def __init__(self):
        # Ultra-low power states (realistic nRF5340 values)
        self.power_states = {
            'system_off': 0.4,          # µW - Complete system shutdown except wake-up circuits
            'deep_sleep': 1.5,          # µW - RAM retention, wake-on-interrupt
            'analog_wake_detector': 2.5, # µW - Hardware analog wake-on-sound detector
            'cpu_startup': 2000,        # µW - Brief CPU startup from deep sleep
            'audio_processing': 45000,  # µW - Full ML inference pipeline
            'radio_tx_low': 4000,       # µW - Low power BLE transmission
            'motion_sensor': 8,         # µW - Accelerometer for movement detection
        }
        
        # Timing parameters (optimized for ultra-low power)
        self.timings = {
            'startup_time_ms': 8,       # Time to wake from system_off
            'inference_time_ms': 10,    # ML inference pipeline
            'radio_tx_time_ms': 3,      # BLE transmission time
            'motion_check_ms': 1,       # Quick motion sensor check
            'wake_decision_ms': 2,      # Hardware wake decision time
        }
        
        # Duty cycling parameters
        self.duty_cycles = {
            'baseline_monitoring': 0.1,      # 10% duty cycle baseline
            'enhanced_monitoring': 0.5,      # 50% when motion detected
            'threat_monitoring': 0.8,        # 80% after threat detection
            'quiet_period_monitoring': 0.02, # 2% during detected quiet periods
        }
        
        logger.info("⚡ Realistic Ultra-Low Power Manager:")
        logger.info(f"   System OFF: {self.power_states['system_off']}µW")
        logger.info(f"   Analog wake detector: {self.power_states['analog_wake_detector']}µW")
        logger.info(f"   Processing time: {self.timings['inference_time_ms']}ms")
    
    def calculate_operational_modes(self):
        """Calculate different operational modes and their power consumption"""
        
        modes = {
            'stealth_mode': {
                'description': 'Minimum power, motion-activated monitoring',
                'base_duty_cycle': 0.02,  # 2% - 30 seconds every 25 minutes
                'motion_activation': True,
                'enhanced_duration_hours': 1,  # 1 hour enhanced monitoring after motion
                'transmission_enabled': False,
                'detection_threshold_db': 55,  # Higher threshold for stealth
            },
            'patrol_mode': {
                'description': 'Balanced power and detection capability',
                'base_duty_cycle': 0.1,   # 10% - 6 seconds every minute
                'motion_activation': True,
                'enhanced_duration_hours': 0.5,  # 30 min enhanced after motion
                'transmission_enabled': True,
                'detection_threshold_db': 50,  # Standard threshold
            },
            'guard_mode': {
                'description': 'Enhanced detection with moderate power',
                'base_duty_cycle': 0.3,   # 30% - 18 seconds every minute
                'motion_activation': False,  # Always monitoring
                'enhanced_duration_hours': 0,
                'transmission_enabled': True,
                'detection_threshold_db': 45,  # Lower threshold, more sensitive
            },
            'emergency_mode': {
                'description': 'Maximum detection capability',
                'base_duty_cycle': 0.8,   # 80% - Nearly continuous
                'motion_activation': False,
                'enhanced_duration_hours': 0,
                'transmission_enabled': True,
                'detection_threshold_db': 40,  # Very sensitive
            }
        }
        
        return modes
    
    def calculate_daily_energy_consumption(self, mode_config, environment_activity_level=0.1):
        """Calculate realistic daily energy consumption"""
        
        # Hours per day in different states
        hours_per_day = 24
        
        # Motion events per day (environment dependent)
        motion_events_per_day = environment_activity_level * 100  # 0.1 = 10 motion events
        
        # Calculate time spent in different monitoring states
        base_monitoring_hours = hours_per_day * mode_config['base_duty_cycle']
        
        if mode_config['motion_activation']:
            # Enhanced monitoring after motion events
            enhanced_hours = min(
                motion_events_per_day * mode_config['enhanced_duration_hours'],
                hours_per_day * 0.8  # Max 80% of day in enhanced mode
            )
            # Remaining time in base monitoring
            base_monitoring_hours = max(0, base_monitoring_hours - enhanced_hours)
            # Time in system OFF
            system_off_hours = hours_per_day - base_monitoring_hours - enhanced_hours
        else:
            enhanced_hours = 0
            system_off_hours = hours_per_day - base_monitoring_hours
        
        # Calculate audio processing events
        # Assume wake events are proportional to monitoring duty cycle
        base_wake_events = base_monitoring_hours * 4  # 4 wake events per hour of monitoring
        enhanced_wake_events = enhanced_hours * 8     # 8 wake events per hour of enhanced monitoring
        total_wake_events = base_wake_events + enhanced_wake_events
        
        # Threat detection events (small fraction)
        threat_events = total_wake_events * 0.02  # 2% are actual threats
        transmission_events = threat_events * 0.7 if mode_config['transmission_enabled'] else 0
        
        # Energy calculations (convert to µJ)
        energy_breakdown = {}
        
        # 1. System OFF energy
        system_off_energy_uj = (self.power_states['system_off'] * system_off_hours * 3600 * 1000) / 1000
        energy_breakdown['system_off'] = system_off_energy_uj
        
        # 2. Base monitoring (analog wake detector + periodic wake)
        analog_detector_energy = (self.power_states['analog_wake_detector'] * base_monitoring_hours * 3600 * 1000) / 1000
        energy_breakdown['analog_detector_base'] = analog_detector_energy
        
        # 3. Enhanced monitoring
        enhanced_analog_energy = (self.power_states['analog_wake_detector'] * enhanced_hours * 3600 * 1000) / 1000
        energy_breakdown['analog_detector_enhanced'] = enhanced_analog_energy
        
        # 4. Motion sensor (when enabled)
        if mode_config['motion_activation']:
            motion_energy = (self.power_states['motion_sensor'] * hours_per_day * 3600 * 1000) / 1000
        else:
            motion_energy = 0
        energy_breakdown['motion_sensor'] = motion_energy
        
        # 5. Wake and processing events
        startup_energy_per_event = (self.power_states['cpu_startup'] * self.timings['startup_time_ms']) / 1000
        processing_energy_per_event = (self.power_states['audio_processing'] * self.timings['inference_time_ms']) / 1000
        
        total_wake_energy = total_wake_events * (startup_energy_per_event + processing_energy_per_event)
        energy_breakdown['wake_and_processing'] = total_wake_energy
        
        # 6. Radio transmissions
        tx_energy_per_event = (self.power_states['radio_tx_low'] * self.timings['radio_tx_time_ms']) / 1000
        transmission_energy = transmission_events * tx_energy_per_event
        energy_breakdown['radio_transmissions'] = transmission_energy
        
        # Total energy
        total_energy_uj = sum(energy_breakdown.values())
        daily_energy_mwh = total_energy_uj / 3600  # Convert µJ to mWh
        
        return {
            'total_energy_uj': total_energy_uj,
            'daily_energy_mwh': daily_energy_mwh,
            'energy_breakdown': energy_breakdown,
            'time_breakdown': {
                'system_off_hours': system_off_hours,
                'base_monitoring_hours': base_monitoring_hours,
                'enhanced_monitoring_hours': enhanced_hours
            },
            'event_counts': {
                'total_wake_events': total_wake_events,
                'threat_events': threat_events,
                'transmission_events': transmission_events,
                'motion_events': motion_events_per_day
            }
        }


class BatteryLifeCalculator:
    """Calculate realistic battery life for different scenarios"""
    
    def __init__(self):
        self.power_manager = RealisticPowerManager()
        
        # Primary lithium battery options
        self.batteries = {
            'standard': {'capacity_mah': 3500, 'chemistry': 'Li-SOCI2'},
            'high_capacity': {'capacity_mah': 5000, 'chemistry': 'Li-SOCI2'},
            'ultra_high': {'capacity_mah': 7000, 'chemistry': 'Li-SOCI2'},
        }
        
        # Deployment scenarios
        self.scenarios = {
            'remote_surveillance': {
                'description': 'Remote area, minimal activity',
                'activity_level': 0.02,  # Very low activity
                'recommended_mode': 'stealth_mode'
            },
            'perimeter_guard': {
                'description': 'Base perimeter, occasional activity',
                'activity_level': 0.1,   # Low activity
                'recommended_mode': 'patrol_mode'
            },
            'urban_patrol': {
                'description': 'Urban area, moderate activity',
                'activity_level': 0.3,   # Moderate activity
                'recommended_mode': 'patrol_mode'
            },
            'active_combat': {
                'description': 'Active combat zone, high activity',
                'activity_level': 0.8,   # High activity
                'recommended_mode': 'guard_mode'
            }
        }
    
    def calculate_battery_life(self, battery_type='high_capacity', scenario='perimeter_guard', mode=None):
        """Calculate battery life for specific configuration"""
        
        battery = self.batteries[battery_type]
        scenario_config = self.scenarios[scenario]
        
        # Get operational mode
        if mode is None:
            mode = scenario_config['recommended_mode']
        
        modes = self.power_manager.calculate_operational_modes()
        mode_config = modes[mode]
        
        # Calculate daily energy consumption
        energy_analysis = self.power_manager.calculate_daily_energy_consumption(
            mode_config, scenario_config['activity_level']
        )
        
        # Battery calculations
        battery_voltage = 3.6  # V
        battery_energy_wh = (battery['capacity_mah'] * battery_voltage) / 1000
        
        # Account for practical factors
        usable_capacity_factor = 0.85  # 85% usable (voltage cutoff, aging)
        temperature_factor = 0.95      # 5% derating for temperature
        self_discharge_factor = 0.98   # 2% per year self-discharge
        
        effective_capacity_wh = battery_energy_wh * usable_capacity_factor * temperature_factor
        
        # Calculate operational time
        daily_energy_wh = energy_analysis['daily_energy_mwh'] / 1000
        days_of_operation = effective_capacity_wh / daily_energy_wh
        
        # Account for self-discharge over time
        if days_of_operation > 365:
            years = days_of_operation / 365
            self_discharge_loss = 1 - (self_discharge_factor ** years)
            days_of_operation *= (1 - self_discharge_loss)
        
        years_of_operation = days_of_operation / 365.25
        
        return {
            'battery_type': battery_type,
            'battery_capacity_mah': battery['capacity_mah'],
            'scenario': scenario,
            'mode': mode,
            'days_of_operation': days_of_operation,
            'years_of_operation': years_of_operation,
            'daily_energy_wh': daily_energy_wh,
            'effective_capacity_wh': effective_capacity_wh,
            'energy_analysis': energy_analysis,
            'mode_config': mode_config
        }
    
    def optimize_for_target_lifetime(self, target_years=1.5):
        """Find configurations that meet target battery life"""
        
        logger.info(f"🎯 Optimizing for {target_years} year target...")
        
        successful_configs = []
        all_results = []
        
        for battery_type in self.batteries.keys():
            for scenario in self.scenarios.keys():
                for mode in self.power_manager.calculate_operational_modes().keys():
                    
                    result = self.calculate_battery_life(battery_type, scenario, mode)
                    all_results.append(result)
                    
                    if result['years_of_operation'] >= target_years:
                        successful_configs.append(result)
                        logger.info(f"   ✅ {battery_type}/{scenario}/{mode}: {result['years_of_operation']:.1f} years")
        
        # Sort successful configs by battery life
        successful_configs.sort(key=lambda x: x['years_of_operation'], reverse=True)
        
        return successful_configs, all_results
    
    def generate_deployment_recommendations(self, target_years=1.5):
        """Generate specific deployment recommendations"""
        
        successful_configs, all_results = self.optimize_for_target_lifetime(target_years)
        
        logger.info(f"\n📊 DEPLOYMENT RECOMMENDATIONS for {target_years} year target:")
        logger.info("=" * 60)
        
        if successful_configs:
            logger.info(f"   ✅ {len(successful_configs)} configurations meet target")
            
            # Best performance
            best = successful_configs[0]
            logger.info(f"\n🏆 BEST PERFORMANCE:")
            logger.info(f"   Battery: {best['battery_capacity_mah']}mAh {self.batteries[best['battery_type']]['chemistry']}")
            logger.info(f"   Scenario: {self.scenarios[best['scenario']]['description']}")
            logger.info(f"   Mode: {best['mode_config']['description']}")
            logger.info(f"   Battery life: {best['years_of_operation']:.1f} years")
            logger.info(f"   Daily energy: {best['daily_energy_wh']*1000:.1f}mWh")
            
            # Practical recommendations by scenario
            logger.info(f"\n📱 PRACTICAL DEPLOYMENT OPTIONS:")
            
            for scenario in self.scenarios.keys():
                scenario_configs = [c for c in successful_configs if c['scenario'] == scenario]
                if scenario_configs:
                    best_for_scenario = scenario_configs[0]
                    logger.info(f"   {scenario}: {best_for_scenario['battery_type']} battery, "
                               f"{best_for_scenario['mode']} mode → {best_for_scenario['years_of_operation']:.1f} years")
            
        else:
            logger.warning(f"   ❌ No configurations meet {target_years} year target")
            
            # Show closest results
            all_results.sort(key=lambda x: x['years_of_operation'], reverse=True)
            logger.info(f"\n🔧 CLOSEST RESULTS:")
            for i, result in enumerate(all_results[:5]):
                logger.info(f"   {i+1}. {result['battery_type']}/{result['scenario']}/{result['mode']}: "
                           f"{result['years_of_operation']:.1f} years")
        
        return successful_configs


def create_ultra_low_power_firmware():
    """Generate ultra-low power firmware configuration"""
    
    logger.info("🔧 Generating ultra-low power firmware configuration...")
    
    power_manager = RealisticPowerManager()
    modes = power_manager.calculate_operational_modes()
    
    firmware_config = {
        "ultra_low_power_system": {
            "hardware_requirements": {
                "analog_wake_detector": {
                    "type": "comparator_based",
                    "power_consumption_uw": 2.5,
                    "detection_threshold_mv": 10,  # Microphone signal threshold
                    "response_time_ms": 1
                },
                "motion_sensor": {
                    "type": "accelerometer_interrupt",
                    "power_consumption_uw": 8,
                    "sensitivity_mg": 50,  # 50mg threshold
                    "sample_rate_hz": 10
                },
                "primary_battery": {
                    "chemistry": "Li-SOCI2",
                    "capacity_mah": 5000,
                    "nominal_voltage_v": 3.6,
                    "cutoff_voltage_v": 2.7
                }
            },
            "power_states": power_manager.power_states,
            "operational_modes": modes,
            "duty_cycling": {
                "wake_interval_base_s": 600,    # 10 minutes baseline
                "wake_duration_ms": 100,        # 100ms wake window
                "motion_enhancement_factor": 5,  # 5x more frequent after motion
                "threat_enhancement_duration_s": 3600  # 1 hour enhanced after threat
            },
            "adaptive_algorithms": {
                "quiet_period_detection": {
                    "enabled": True,
                    "threshold_db": 30,
                    "min_duration_min": 30,
                    "duty_cycle_reduction": 0.5  # Reduce to 50% duty cycle
                },
                "activity_based_adjustment": {
                    "enabled": True,
                    "motion_threshold_mg": 50,
                    "enhancement_duration_min": 60
                },
                "battery_conservation": {
                    "enabled": True,
                    "low_battery_threshold_percent": 20,
                    "critical_battery_threshold_percent": 5,
                    "conservation_mode_duty_reduction": 0.3
                }
            }
        },
        "performance_maintenance": {
            "ml_model_optimizations": {
                "quantization": "Q7",
                "sparsity_level": 0.9,
                "inference_time_budget_ms": 10,
                "accuracy_target_percent": 95
            },
            "threat_detection": {
                "confidence_threshold": 0.6,
                "false_positive_rate_target": 0.02,
                "detection_latency_budget_ms": 15
            },
            "communication": {
                "transmission_power_dbm": -12,  # Low power for local mesh
                "mesh_hop_limit": 3,
                "transmission_on_threat_only": True,
                "heartbeat_interval_hours": 24
            }
        },
        "deployment_configurations": {
            "stealth_surveillance": {
                "target_battery_life_years": 2.5,
                "duty_cycle_percent": 2,
                "motion_activation": True,
                "transmission_enabled": False
            },
            "perimeter_guard": {
                "target_battery_life_years": 1.5,
                "duty_cycle_percent": 10,
                "motion_activation": True,
                "transmission_enabled": True
            },
            "urban_patrol": {
                "target_battery_life_years": 1.0,
                "duty_cycle_percent": 30,
                "motion_activation": True,
                "transmission_enabled": True
            }
        }
    }
    
    # Save firmware configuration
    config_path = Path('sait_01_firmware/realistic_ultra_low_power_config.json')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(firmware_config, f, indent=2)
    
    logger.info(f"✅ Ultra-low power firmware config saved: {config_path}")
    
    return firmware_config


def main():
    """Main ultra-low power optimization"""
    
    logger.info("🔋 Realistic Ultra-Low Power Optimization")
    logger.info("Enhanced QADT-R for 1-2 Year Battery Life")
    logger.info("=" * 60)
    
    # Initialize battery life calculator
    calculator = BatteryLifeCalculator()
    
    # Test specific promising configurations
    logger.info("\n📊 Testing Promising Configurations:")
    
    test_configs = [
        ('high_capacity', 'remote_surveillance', 'stealth_mode'),
        ('high_capacity', 'perimeter_guard', 'patrol_mode'),
        ('ultra_high', 'remote_surveillance', 'stealth_mode'),
        ('ultra_high', 'perimeter_guard', 'patrol_mode'),
    ]
    
    results = []
    for battery, scenario, mode in test_configs:
        result = calculator.calculate_battery_life(battery, scenario, mode)
        results.append(result)
        
        logger.info(f"   {battery}/{scenario}/{mode}:")
        logger.info(f"     Battery life: {result['years_of_operation']:.1f} years")
        logger.info(f"     Daily energy: {result['daily_energy_wh']*1000:.1f}mWh")
    
    # Generate deployment recommendations
    successful_configs = calculator.generate_deployment_recommendations(target_years=1.5)
    
    # Generate firmware configuration
    firmware_config = create_ultra_low_power_firmware()
    
    # Save results
    with open('realistic_ultra_low_power_results.json', 'w') as f:
        json.dump({
            'test_results': results,
            'successful_configs': successful_configs,
            'firmware_config': firmware_config
        }, f, indent=2, default=str)
    
    # Final assessment
    logger.info(f"\n🏆 ULTRA-LOW POWER OPTIMIZATION RESULTS:")
    logger.info("=" * 50)
    
    if successful_configs:
        best = successful_configs[0]
        logger.info(f"✅ TARGET ACHIEVED: {best['years_of_operation']:.1f} years battery life")
        logger.info(f"📋 Configuration: {best['battery_capacity_mah']}mAh battery")
        logger.info(f"📍 Best scenario: {best['scenario']}")
        logger.info(f"⚙️  Operating mode: {best['mode']}")
        logger.info(f"⚡ Daily energy: {best['daily_energy_wh']*1000:.1f}mWh")
        
        logger.info(f"\n🎯 KEY OPTIMIZATIONS:")
        logger.info(f"   🔋 Primary Li-SOCI2 battery (5000-7000mAh)")
        logger.info(f"   👂 Hardware analog wake-on-sound (2.5µW)")
        logger.info(f"   🏃 Motion-activated enhanced monitoring")
        logger.info(f"   ⏰ Aggressive duty cycling (2-10%)")
        logger.info(f"   📡 Transmission only on threats")
        logger.info(f"   🛌 System OFF between monitoring windows")
        
        logger.info(f"\n🚀 DEPLOYMENT READY FOR 1-2 YEAR OPERATION")
        return True
    
    else:
        logger.warning(f"⚠️ 1.5 year target not achieved with current design")
        logger.info(f"🔧 Consider additional optimizations:")
        logger.info(f"   - Larger battery capacity (>7000mAh)")
        logger.info(f"   - More aggressive duty cycling")
        logger.info(f"   - Application-specific monitoring patterns")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)