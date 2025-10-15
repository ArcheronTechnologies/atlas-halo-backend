# 🧹 Codebase Cleanup Plan

## Current Production Files (KEEP)
**Essential for Phase 3 and deployment:**

### Core Training & Model Files
- `train_enhanced_qadt_r_with_drones.py` - Final enhanced training with drone integration
- `advanced_qadt_r.py` - Core QADT-R architecture 
- `adaptive_unified_defense.py` - Advanced defense mechanisms

### Hardware Integration & Optimization
- `aggressive_model_compression.py` - Final compression pipeline
- `convert_pytorch_to_cmsis.py` - nRF5340 deployment pipeline
- `audio_processing_pipeline.py` - Production audio processing

### Validation & Testing
- `phase1_2_comprehensive_validation.py` - Complete Phase 1-2 validation
- `environmental_stress_testing.py` - Environmental testing framework
- `battlefield_validation_test.py` - Battlefield performance validation
- `realistic_ultra_low_power.py` - Ultra-low power optimization

### Firmware & Integration
- `sait_01_firmware/` directory - Complete firmware stack
- Model weight files and CMSIS-NN outputs

---

## Files to DELETE (Outdated/Superseded)

### Early Development Iterations
```bash
# Phase 1 iterations (superseded by phase1_2_comprehensive_validation.py)
rm phase1_5_comprehensive_validation.py
rm phase1_6_improved_training.py
rm phase1_6_military_training.py
rm test_phase1_simplified.py
rm test_phase1_complete.py
rm test_phase1_comprehensive.py

# Phase 2.1 iterations (superseded by train_enhanced_qadt_r_with_drones.py)
rm "phase_2_1_complete_collection 2.py"
rm phase_2_1_complete_collection.py
rm phase_2_1_deployment.py
rm phase_2_2_complete_training.py
rm phase_2_2_fixed.py
rm phase_2_2_real_world_training.py
```

### Architecture Experiments (Superseded)
```bash
# Early architecture attempts (superseded by advanced_qadt_r.py)
rm enhanced_architecture_fixed.py
rm improved_architecture.py
rm lightweight_enhanced_architecture.py
rm noise_robust_architecture.py
rm enhanced_model_training.py

# Accuracy experiments (integrated into final training)
rm accuracy_improvement_analysis.py
rm final_accuracy_boost.py
rm comprehensive_fixes_validation.py
```

### Data Pipeline Experiments
```bash
# Early data pipeline attempts (superseded by train_enhanced_qadt_r_with_drones.py)
rm fixed_data_pipeline.py
rm domain_adaptation_training.py
rm domain_gap_analysis.py
rm physics_based_training.py
rm physics_training_fixed.py
rm real_world_audio_collection.py
rm real_audio_downloader.py
```

### Test Files (Early Development)
```bash
# Individual component tests (superseded by comprehensive validation)
rm test_battlefield_model.py
rm test_prototypical_contrastive.py
rm test_qadt_r_proper.py
rm test_qadt_r_simple.py
rm test_qadt_r_comprehensive.py
rm test_complete_integration.py
rm test_enhanced_training.py
rm test_military_model.py
rm test_adversarial_defense.py
rm test_adversarial_defense_simple.py
rm test_noise_robust_battlefield.py
rm test_power_optimization.py

# Quick fixes and minimal tests
rm quick_validation_test.py
rm quick_model_fix.py
rm minimal_fix.py
```

### Adversarial Defense Experiments
```bash
# Early adversarial defense (integrated into adaptive_unified_defense.py)
rm adversarial_defense_comprehensive_test.py
rm adversarial_defense_performance_improvements.py
rm memory_based_defense.py
```

### Military-Specific Tests (Early)
```bash
# Early military tests (superseded by comprehensive validation)
rm test_military_attack_patterns.py
rm test_electronic_warfare.py
rm test_tiered_robustness.py
rm test_military_defense_integration.py
rm test_military_multiclass_training.py
```

### Compression Experiments
```bash
# Early compression attempts (superseded by aggressive_model_compression.py)
rm compress_enhanced_model.py
rm run_aggressive_compression.py
rm validate_enhanced_cmsis_deployment.py
```

### Audio Processing Experiments
```bash
# Audio experiments (superseded by audio_processing_pipeline.py)
rm environmental_acoustic_mapping.py
rm nato_acoustic_enhancements.py
rm robust_augmentation_pipeline.py
```

### Optimization Experiments
```bash
# Early optimization attempts (superseded by realistic_ultra_low_power.py)
rm run_bayesian_optimization.py
rm ultra_low_power_optimization.py
```

### Hardware Test Files
```bash
# Early hardware tests (superseded by comprehensive validation)
rm phase3_hardware_integration_test.py
```

### Validation Experiments
```bash
# Early validation attempts (superseded by phase1_2_comprehensive_validation.py)
rm battlefield_validation.py
rm real_world_validation.py
```

### Downloader Scripts (No longer needed)
```bash
rm download_real_mad_dataset.py
```

### Miscellaneous Development Files
```bash
rm test_aaa_framework.py  # AAA superseded by integrated training
rm test_tinyml_accuracy.py  # Accuracy testing integrated
```

---

## JSON Files to Review

### Training History Files
- Keep: `enhanced_training_history.json` (final training results)
- Delete: Early experiment history files

### Model Artifacts
- Keep: Final compressed model files
- Delete: Intermediate compression experiments

---

## Summary
- **Total Python files:** 87
- **Files to keep:** ~15 core production files
- **Files to delete:** ~72 outdated development files
- **Space savings:** Significant reduction in codebase complexity
- **Benefit:** Cleaner structure for Phase 3 development

This cleanup removes 80%+ of development iteration files while preserving all production-ready components.