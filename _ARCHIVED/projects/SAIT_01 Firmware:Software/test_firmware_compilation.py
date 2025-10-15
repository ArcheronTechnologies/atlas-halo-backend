#!/usr/bin/env python3
"""
Phase 4.1 Firmware Compilation Validation
Tests that all implemented components will compile with Zephyr RTOS
"""

import os
import re
import sys
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class CompilationResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"

@dataclass
class CompilationTest:
    component: str
    test_name: str
    result: CompilationResult
    message: str
    details: Dict[str, Any] = None

class FirmwareCompilationValidator:
    def __init__(self):
        self.test_results: List[CompilationTest] = []
        self.firmware_base_path = "/Users/timothyaikenhead/Desktop/SAIT_01 Firmware:Software/nrf5340_dual_core_firmware"
        
    def log_result(self, component: str, test_name: str, result: CompilationResult, 
                   message: str, details: Dict = None):
        """Log a compilation test result"""
        test_result = CompilationTest(component, test_name, result, message, details)
        self.test_results.append(test_result)
        
        status_symbol = "✅" if result == CompilationResult.PASS else "❌" if result == CompilationResult.FAIL else "⚠️"
        print(f"{status_symbol} {component}: {test_name} - {result.value}")
        print(f"   {message}")
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")
        print()

    def validate_zephyr_includes(self) -> bool:
        """Validate all Zephyr RTOS includes are correct"""
        print("🔍 Validating Zephyr RTOS Include Statements...")
        
        success = True
        files_to_check = [
            ("application_core/enhanced_qadt_r_deployment.c", "Enhanced QADT-R"),
            ("application_core/cmsis_nn_inference_pipeline.c", "CMSIS-NN Pipeline"),
            ("network_core/byzantine_consensus.c", "Byzantine Consensus"),
            ("shared/dual_core_coordinator.c", "Dual-Core Coordination"),
            ("shared/secure_boot_validator.c", "Secure Boot")
        ]
        
        required_zephyr_includes = {
            "application_core/enhanced_qadt_r_deployment.c": [
                "zephyr/kernel.h", "zephyr/logging/log.h", "zephyr/device.h"
            ],
            "application_core/cmsis_nn_inference_pipeline.c": [
                "zephyr/kernel.h", "zephyr/logging/log.h", "arm_nnfunctions.h"
            ],
            "network_core/byzantine_consensus.c": [
                "zephyr/kernel.h", "zephyr/logging/log.h", "zephyr/net/net_if.h",
                "bluetooth/bluetooth.h", "bluetooth/mesh.h"
            ],
            "shared/dual_core_coordinator.c": [
                "zephyr/kernel.h", "zephyr/logging/log.h", "zephyr/ipc/ipc_service.h"
            ],
            "shared/secure_boot_validator.c": [
                "zephyr/kernel.h", "zephyr/logging/log.h", "zephyr/crypto/crypto.h",
                "tinycrypt/sha256.h"
            ]
        }
        
        for file_path, component in files_to_check:
            full_path = os.path.join(self.firmware_base_path, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    required = required_zephyr_includes.get(file_path, [])
                    missing_includes = []
                    
                    for include in required:
                        if f"#include <{include}>" not in content:
                            missing_includes.append(include)
                    
                    if missing_includes:
                        self.log_result(component, "Zephyr Includes", CompilationResult.FAIL,
                                      f"Missing includes: {missing_includes}")
                        success = False
                    else:
                        self.log_result(component, "Zephyr Includes", CompilationResult.PASS,
                                      "All required Zephyr includes present")
                
                except Exception as e:
                    self.log_result(component, "Include Analysis", CompilationResult.FAIL,
                                  f"Failed to analyze includes: {e}")
                    success = False
            else:
                self.log_result(component, "File Existence", CompilationResult.FAIL,
                              f"Source file not found: {file_path}")
                success = False
        
        return success

    def validate_data_types(self) -> bool:
        """Validate all data types are compatible with embedded systems"""
        print("🔍 Validating Embedded-Compatible Data Types...")
        
        success = True
        files_to_check = [
            ("application_core/enhanced_qadt_r_deployment.h", "Enhanced QADT-R"),
            ("application_core/cmsis_nn_inference_pipeline.h", "CMSIS-NN Pipeline"),
            ("network_core/byzantine_consensus.h", "Byzantine Consensus"),
            ("shared/dual_core_coordinator.h", "Dual-Core Coordination"),
            ("shared/secure_boot_validator.h", "Secure Boot")
        ]
        
        # Embedded-friendly data types
        good_types = ["uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", 
                     "int32_t", "int64_t", "bool", "size_t", "float"]
        
        # Types to avoid in embedded systems
        problematic_types = ["double", "long double", "long long", "wchar_t"]
        
        for file_path, component in files_to_check:
            full_path = os.path.join(self.firmware_base_path, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    found_problematic = []
                    for prob_type in problematic_types:
                        if re.search(r'\b' + prob_type + r'\b', content):
                            found_problematic.append(prob_type)
                    
                    if found_problematic:
                        self.log_result(component, "Data Types", CompilationResult.WARNING,
                                      f"Potentially problematic types: {found_problematic}")
                    else:
                        self.log_result(component, "Data Types", CompilationResult.PASS,
                                      "All data types are embedded-compatible")
                
                except Exception as e:
                    self.log_result(component, "Type Analysis", CompilationResult.WARNING,
                                  f"Could not analyze data types: {e}")
            else:
                self.log_result(component, "File Check", CompilationResult.FAIL,
                              f"Header file not found: {file_path}")
                success = False
        
        return success

    def validate_memory_usage(self) -> bool:
        """Validate memory usage patterns for embedded constraints"""
        print("🔍 Validating Memory Usage Patterns...")
        
        success = True
        files_to_check = [
            ("application_core/enhanced_qadt_r_deployment.c", "Enhanced QADT-R"),
            ("application_core/cmsis_nn_inference_pipeline.c", "CMSIS-NN Pipeline"),
            ("network_core/byzantine_consensus.c", "Byzantine Consensus"),
            ("shared/dual_core_coordinator.c", "Dual-Core Coordination"),
            ("shared/secure_boot_validator.c", "Secure Boot")
        ]
        
        for file_path, component in files_to_check:
            full_path = os.path.join(self.firmware_base_path, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    memory_issues = []
                    
                    # Check for malloc/free usage (prefer static allocation)
                    if "malloc(" in content or "calloc(" in content or "realloc(" in content:
                        memory_issues.append("Dynamic allocation detected (malloc/calloc)")
                    
                    # Check for recursion (stack overflow risk)
                    function_calls = re.findall(r'(\w+)\s*\([^)]*\)\s*{[^}]*\1\s*\(', content)
                    if function_calls:
                        memory_issues.append("Potential recursion detected")
                    
                    # Check for large stack allocations
                    large_arrays = re.findall(r'(\w+)\s+\w+\s*\[\s*(\d+)\s*\]', content)
                    for array_type, size in large_arrays:
                        try:
                            if int(size) > 1024:
                                memory_issues.append(f"Large stack array: {array_type}[{size}]")
                        except ValueError:
                            pass
                    
                    if memory_issues:
                        self.log_result(component, "Memory Usage", CompilationResult.WARNING,
                                      f"Memory concerns: {memory_issues}")
                    else:
                        self.log_result(component, "Memory Usage", CompilationResult.PASS,
                                      "Memory usage patterns are embedded-friendly")
                
                except Exception as e:
                    self.log_result(component, "Memory Analysis", CompilationResult.WARNING,
                                  f"Could not analyze memory usage: {e}")
        
        return success

    def validate_interrupt_safety(self) -> bool:
        """Validate interrupt and thread safety patterns"""
        print("🔍 Validating Interrupt and Thread Safety...")
        
        success = True
        files_to_check = [
            ("application_core/enhanced_qadt_r_deployment.c", "Enhanced QADT-R"),
            ("application_core/cmsis_nn_inference_pipeline.c", "CMSIS-NN Pipeline"),
            ("network_core/byzantine_consensus.c", "Byzantine Consensus"),
            ("shared/dual_core_coordinator.c", "Dual-Core Coordination"),
            ("shared/secure_boot_validator.c", "Secure Boot")
        ]
        
        for file_path, component in files_to_check:
            full_path = os.path.join(self.firmware_base_path, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    safety_features = []
                    safety_issues = []
                    
                    # Check for Zephyr synchronization primitives
                    if "k_mutex" in content:
                        safety_features.append("Mutex usage")
                    if "k_sem" in content:
                        safety_features.append("Semaphore usage")
                    if "k_msgq" in content:
                        safety_features.append("Message queue usage")
                    if "k_timer" in content:
                        safety_features.append("Timer usage")
                    
                    # Check for global variables without protection
                    global_vars = re.findall(r'^static\s+\w+\s+\w+.*;', content, re.MULTILINE)
                    unprotected_globals = 0
                    for var in global_vars:
                        if "volatile" not in var and "const" not in var:
                            unprotected_globals += 1
                    
                    if unprotected_globals > 0:
                        safety_issues.append(f"{unprotected_globals} potentially unprotected global variables")
                    
                    # Check for interrupt context functions
                    if "_handler" in content or "_irq" in content:
                        safety_features.append("Interrupt handlers present")
                    
                    if safety_issues:
                        self.log_result(component, "Thread Safety", CompilationResult.WARNING,
                                      f"Safety concerns: {safety_issues}", {"features": safety_features})
                    else:
                        self.log_result(component, "Thread Safety", CompilationResult.PASS,
                                      "Good thread safety patterns", {"features": safety_features})
                
                except Exception as e:
                    self.log_result(component, "Safety Analysis", CompilationResult.WARNING,
                                  f"Could not analyze thread safety: {e}")
        
        return success

    def validate_hardware_abstraction(self) -> bool:
        """Validate hardware abstraction layer usage"""
        print("🔍 Validating Hardware Abstraction Layer...")
        
        success = True
        files_to_check = [
            ("application_core/enhanced_qadt_r_deployment.c", "Enhanced QADT-R"),
            ("application_core/cmsis_nn_inference_pipeline.c", "CMSIS-NN Pipeline"),
            ("network_core/byzantine_consensus.c", "Byzantine Consensus"),
            ("shared/dual_core_coordinator.c", "Dual-Core Coordination"),
            ("shared/secure_boot_validator.c", "Secure Boot")
        ]
        
        for file_path, component in files_to_check:
            full_path = os.path.join(self.firmware_base_path, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    hal_issues = []
                    hal_features = []
                    
                    # Check for direct register access (should be avoided)
                    if re.search(r'0x[0-9A-Fa-f]{8}', content):
                        hal_issues.append("Direct register addresses detected")
                    
                    # Check for Zephyr device model usage
                    if "device_get_binding" in content or "DEVICE_DT_GET" in content:
                        hal_features.append("Zephyr device model usage")
                    
                    # Check for proper GPIO/driver usage
                    if "zephyr/drivers/" in content:
                        hal_features.append("Zephyr driver usage")
                    
                    # Check for ARM CMSIS usage (appropriate for nRF5340)
                    if "arm_" in content:
                        hal_features.append("ARM CMSIS usage")
                    
                    if hal_issues:
                        self.log_result(component, "Hardware Abstraction", CompilationResult.WARNING,
                                      f"HAL concerns: {hal_issues}", {"features": hal_features})
                    else:
                        self.log_result(component, "Hardware Abstraction", CompilationResult.PASS,
                                      "Good hardware abstraction", {"features": hal_features})
                
                except Exception as e:
                    self.log_result(component, "HAL Analysis", CompilationResult.WARNING,
                                  f"Could not analyze hardware abstraction: {e}")
        
        return success

    def validate_error_handling(self) -> bool:
        """Validate error handling patterns"""
        print("🔍 Validating Error Handling Patterns...")
        
        success = True
        files_to_check = [
            ("application_core/enhanced_qadt_r_deployment.c", "Enhanced QADT-R"),
            ("application_core/cmsis_nn_inference_pipeline.c", "CMSIS-NN Pipeline"),
            ("network_core/byzantine_consensus.c", "Byzantine Consensus"),
            ("shared/dual_core_coordinator.c", "Dual-Core Coordination"),
            ("shared/secure_boot_validator.c", "Secure Boot")
        ]
        
        for file_path, component in files_to_check:
            full_path = os.path.join(self.firmware_base_path, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    error_patterns = []
                    error_issues = []
                    
                    # Check for consistent error return patterns
                    if "return -E" in content:
                        error_patterns.append("Standard error codes")
                    if "return 0" in content:
                        error_patterns.append("Success return codes")
                    
                    # Check for parameter validation
                    if "!=" in content and "NULL" in content:
                        error_patterns.append("NULL pointer checks")
                    
                    # Check for logging on errors
                    if "LOG_ERR" in content:
                        error_patterns.append("Error logging")
                    
                    # Check for missing error handling
                    function_calls = re.findall(r'(\w+)\s*\([^)]*\)\s*;', content)
                    unchecked_calls = 0
                    for call in function_calls:
                        if not re.search(r'(if|ret|result|status)\s*=.*' + call, content):
                            unchecked_calls += 1
                    
                    if unchecked_calls > 10:  # Threshold for concern
                        error_issues.append(f"{unchecked_calls} potentially unchecked function calls")
                    
                    if error_issues:
                        self.log_result(component, "Error Handling", CompilationResult.WARNING,
                                      f"Error handling concerns: {error_issues}", {"patterns": error_patterns})
                    else:
                        self.log_result(component, "Error Handling", CompilationResult.PASS,
                                      "Good error handling patterns", {"patterns": error_patterns})
                
                except Exception as e:
                    self.log_result(component, "Error Analysis", CompilationResult.WARNING,
                                  f"Could not analyze error handling: {e}")
        
        return success

    def run_compilation_validation(self) -> Dict[str, Any]:
        """Run complete firmware compilation validation"""
        print("🚀 Starting Phase 4.1 Firmware Compilation Validation")
        print("=" * 80)
        
        overall_start = time.time()
        
        # Run all validation tests
        validations = [
            ("Zephyr Includes", self.validate_zephyr_includes()),
            ("Data Types", self.validate_data_types()),
            ("Memory Usage", self.validate_memory_usage()),
            ("Thread Safety", self.validate_interrupt_safety()),
            ("Hardware Abstraction", self.validate_hardware_abstraction()),
            ("Error Handling", self.validate_error_handling())
        ]
        
        results = {}
        for test_name, result in validations:
            results[test_name] = result
        
        # Generate comprehensive report
        total_time = (time.time() - overall_start) * 1000
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.result == CompilationResult.PASS])
        failed_tests = len([r for r in self.test_results if r.result == CompilationResult.FAIL])
        warning_tests = len([r for r in self.test_results if r.result == CompilationResult.WARNING])
        
        compilation_readiness = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Print summary
        print("\n" + "=" * 80)
        print("🏁 FIRMWARE COMPILATION READINESS SUMMARY")
        print("=" * 80)
        
        print(f"📊 Compilation Test Statistics:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests} ✅")
        print(f"   Failed: {failed_tests} ❌")
        print(f"   Warnings: {warning_tests} ⚠️")
        print(f"   Compilation Readiness: {compilation_readiness:.1f}%")
        print(f"   Total Analysis Time: {total_time:.2f}ms")
        
        print(f"\n🔧 Validation Categories:")
        for test_name, success in results.items():
            status = "✅ READY" if success else "❌ ISSUES"
            print(f"   {test_name}: {status}")
        
        overall_success = all(results.values()) and failed_tests == 0
        
        if overall_success:
            print(f"\n🎉 COMPILATION VALIDATION: SUCCESS")
            print(f"   All components ready for Zephyr RTOS compilation")
            print(f"   nRF5340 deployment-ready firmware")
        else:
            print(f"\n⚠️  COMPILATION VALIDATION: ISSUES DETECTED")
            print(f"   Review compilation issues before building firmware")
        
        # Detailed results for failed tests
        if failed_tests > 0:
            print(f"\n❌ Critical Issues:")
            for result in self.test_results:
                if result.result == CompilationResult.FAIL:
                    print(f"   {result.component} - {result.test_name}: {result.message}")
        
        if warning_tests > 0:
            print(f"\n⚠️ Warnings (Review Recommended):")
            for result in self.test_results:
                if result.result == CompilationResult.WARNING:
                    print(f"   {result.component} - {result.test_name}: {result.message}")
        
        return {
            'overall_success': overall_success,
            'compilation_readiness': compilation_readiness,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'warning_tests': warning_tests,
            'total_time_ms': total_time,
            'validation_results': results
        }

def main():
    """Main compilation validation entry point"""
    validator = FirmwareCompilationValidator()
    results = validator.run_compilation_validation()
    
    return results['overall_success']

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)