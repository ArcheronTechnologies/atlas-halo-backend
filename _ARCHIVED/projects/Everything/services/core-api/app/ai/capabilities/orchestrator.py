"""
Capability Orchestrator

Orchestrates multiple AI capabilities into complex pipelines with 
parallel execution, fallbacks, and data flow management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json

from .base import AICapability, CapabilityResult, CapabilityStatus
from .registry import CapabilityRegistry, capability_registry

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A step in a capability pipeline"""
    capability_name: str
    input_mapping: Optional[Dict[str, str]] = None  # Map pipeline data to capability input
    output_mapping: Optional[Dict[str, str]] = None  # Map capability output to pipeline data
    condition: Optional[str] = None  # Condition for execution (e.g., "confidence > 0.8")
    parallel_group: Optional[str] = None  # Group name for parallel execution
    timeout_seconds: Optional[float] = None
    required: bool = True  # Whether step failure should fail the entire pipeline


@dataclass 
class PipelineConfig:
    """Configuration for a capability pipeline"""
    name: str
    version: str
    steps: List[PipelineStep]
    timeout_seconds: float = 120.0
    parallel_execution: bool = True
    fail_fast: bool = False  # Stop on first failure
    retry_failed_steps: bool = True
    output_schema: Optional[Dict[str, Any]] = None


@dataclass
class PipelineResult:
    """Result from pipeline execution"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, CapabilityResult] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    pipeline_name: Optional[str] = None
    pipeline_version: Optional[str] = None


class Pipeline:
    """
    A pipeline of AI capabilities that can be executed with data flow orchestration.
    """
    
    def __init__(self, config: PipelineConfig, registry: Optional[CapabilityRegistry] = None):
        self.config = config
        self.registry = registry or capability_registry
        self._pipeline_data = {}
        
    async def execute(self, initial_data: Dict[str, Any]) -> PipelineResult:
        """
        Execute the pipeline with the given initial data.
        
        Args:
            initial_data: Initial data to feed into the pipeline
            
        Returns:
            PipelineResult with execution results
        """
        start_time = asyncio.get_event_loop().time()
        self._pipeline_data = initial_data.copy()
        step_results = {}
        warnings = []
        
        try:
            logger.info(f"Executing pipeline: {self.config.name}")
            
            # Group steps by parallel groups
            step_groups = self._group_steps_for_execution()
            
            # Execute step groups
            for group_name, steps in step_groups:
                if self.config.parallel_execution and len(steps) > 1:
                    # Execute steps in parallel
                    group_results = await self._execute_parallel_steps(steps)
                else:
                    # Execute steps sequentially
                    group_results = await self._execute_sequential_steps(steps)
                
                # Merge results
                step_results.update(group_results)
                
                # Check for failures if fail_fast is enabled
                if self.config.fail_fast:
                    failed_steps = [
                        name for name, result in group_results.items() 
                        if not result.success
                    ]
                    if failed_steps:
                        return PipelineResult(
                            success=False,
                            data=self._pipeline_data,
                            step_results=step_results,
                            error=f"Pipeline failed fast due to failed steps: {failed_steps}",
                            pipeline_name=self.config.name,
                            pipeline_version=self.config.version
                        )
            
            # Calculate execution time
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Check overall success
            failed_required_steps = [
                step.capability_name for step in self.config.steps
                if step.required and not step_results.get(step.capability_name, CapabilityResult(success=False)).success
            ]
            
            success = len(failed_required_steps) == 0
            
            if not success:
                warnings.append(f"Required steps failed: {failed_required_steps}")
            
            return PipelineResult(
                success=success,
                data=self._pipeline_data,
                step_results=step_results,
                execution_time_ms=execution_time,
                warnings=warnings,
                pipeline_name=self.config.name,
                pipeline_version=self.config.version
            )
            
        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.error(f"Pipeline execution error: {e}", exc_info=True)
            
            return PipelineResult(
                success=False,
                data=self._pipeline_data,
                step_results=step_results,
                execution_time_ms=execution_time,
                error=str(e),
                pipeline_name=self.config.name,
                pipeline_version=self.config.version
            )
    
    def _group_steps_for_execution(self) -> List[tuple[str, List[PipelineStep]]]:
        """Group steps for parallel/sequential execution"""
        groups = []
        current_group = []
        current_group_name = None
        
        for step in self.config.steps:
            if step.parallel_group:
                if step.parallel_group != current_group_name:
                    # Start new group
                    if current_group:
                        groups.append((current_group_name or "sequential", current_group))
                    current_group = [step]
                    current_group_name = step.parallel_group
                else:
                    # Add to current group
                    current_group.append(step)
            else:
                # Sequential step
                if current_group:
                    groups.append((current_group_name or "sequential", current_group))
                groups.append(("sequential", [step]))
                current_group = []
                current_group_name = None
        
        # Add final group
        if current_group:
            groups.append((current_group_name or "sequential", current_group))
        
        return groups
    
    async def _execute_parallel_steps(self, steps: List[PipelineStep]) -> Dict[str, CapabilityResult]:
        """Execute multiple steps in parallel"""
        tasks = []
        step_names = []
        
        for step in steps:
            if self._should_execute_step(step):
                task = self._execute_single_step(step)
                tasks.append(task)
                step_names.append(step.capability_name)
        
        if not tasks:
            return {}
        
        logger.debug(f"Executing {len(tasks)} steps in parallel: {step_names}")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        step_results = {}
        for i, (step_name, result) in enumerate(zip(step_names, results)):
            if isinstance(result, Exception):
                step_results[step_name] = CapabilityResult(
                    success=False,
                    error=str(result),
                    capability_name=step_name
                )
            else:
                step_results[step_name] = result
        
        return step_results
    
    async def _execute_sequential_steps(self, steps: List[PipelineStep]) -> Dict[str, CapabilityResult]:
        """Execute steps sequentially"""
        step_results = {}
        
        for step in steps:
            if self._should_execute_step(step):
                step_name = step.capability_name
                logger.debug(f"Executing step: {step_name}")
                
                result = await self._execute_single_step(step)
                step_results[step_name] = result
                
                # Stop if this is a required step that failed
                if step.required and not result.success and self.config.fail_fast:
                    break
        
        return step_results
    
    async def _execute_single_step(self, step: PipelineStep) -> CapabilityResult:
        """Execute a single pipeline step"""
        capability = self.registry.get(step.capability_name)
        
        if not capability:
            return CapabilityResult(
                success=False,
                error=f"Capability '{step.capability_name}' not found in registry",
                capability_name=step.capability_name
            )
        
        # Map input data
        step_input = self._map_input_data(step)
        
        try:
            # Execute capability with optional timeout
            if step.timeout_seconds:
                result = await asyncio.wait_for(
                    capability.run(step_input),
                    timeout=step.timeout_seconds
                )
            else:
                result = await capability.run(step_input)
            
            # Map output data back to pipeline
            if result.success and result.data:
                self._map_output_data(step, result.data)
            
            return result
            
        except asyncio.TimeoutError:
            return CapabilityResult(
                success=False,
                error=f"Step '{step.capability_name}' timed out after {step.timeout_seconds}s",
                capability_name=step.capability_name
            )
        except Exception as e:
            logger.error(f"Error executing step {step.capability_name}: {e}", exc_info=True)
            return CapabilityResult(
                success=False,
                error=str(e),
                capability_name=step.capability_name
            )
    
    def _should_execute_step(self, step: PipelineStep) -> bool:
        """Check if a step should be executed based on its condition"""
        if not step.condition:
            return True
        
        try:
            # Simple condition evaluation (could be enhanced with a proper expression evaluator)
            # For now, support basic comparisons like "confidence > 0.8"
            condition = step.condition.strip()
            
            # Replace variables with actual values
            for key, value in self._pipeline_data.items():
                if isinstance(value, (int, float)):
                    condition = condition.replace(key, str(value))
                elif isinstance(value, str):
                    condition = condition.replace(key, f"'{value}'")
            
            # Evaluate the condition safely
            return eval(condition)
            
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{step.condition}': {e}")
            return True  # Default to executing the step
    
    def _map_input_data(self, step: PipelineStep) -> Dict[str, Any]:
        """Map pipeline data to capability input"""
        if not step.input_mapping:
            return self._pipeline_data.copy()
        
        mapped_data = {}
        for input_key, pipeline_key in step.input_mapping.items():
            if pipeline_key in self._pipeline_data:
                mapped_data[input_key] = self._pipeline_data[pipeline_key]
        
        return mapped_data
    
    def _map_output_data(self, step: PipelineStep, output_data: Any) -> None:
        """Map capability output back to pipeline data"""
        if not step.output_mapping:
            # If no mapping specified, merge the output data if it's a dict
            if isinstance(output_data, dict):
                self._pipeline_data.update(output_data)
            else:
                self._pipeline_data[f"{step.capability_name}_result"] = output_data
            return
        
        if isinstance(output_data, dict):
            for output_key, pipeline_key in step.output_mapping.items():
                if output_key in output_data:
                    self._pipeline_data[pipeline_key] = output_data[output_key]
        else:
            # Single value output
            if len(step.output_mapping) == 1:
                pipeline_key = list(step.output_mapping.values())[0]
                self._pipeline_data[pipeline_key] = output_data


class CapabilityOrchestrator:
    """
    High-level orchestrator for AI capabilities and pipelines.
    
    Provides simplified interfaces for common use cases and manages
    the execution of complex capability workflows.
    """
    
    def __init__(self, registry: Optional[CapabilityRegistry] = None):
        self.registry = registry or capability_registry
        self._pipelines: Dict[str, Pipeline] = {}
    
    def register_pipeline(self, config: PipelineConfig) -> None:
        """Register a pipeline configuration"""
        pipeline = Pipeline(config, self.registry)
        self._pipelines[config.name] = pipeline
        logger.info(f"Registered pipeline: {config.name}")
    
    async def execute_pipeline(self, name: str, data: Dict[str, Any]) -> PipelineResult:
        """Execute a registered pipeline by name"""
        if name not in self._pipelines:
            return PipelineResult(
                success=False,
                error=f"Pipeline '{name}' not found",
                pipeline_name=name
            )
        
        return await self._pipelines[name].execute(data)
    
    async def execute_capability(
        self, 
        name: str, 
        data: Dict[str, Any],
        fallback_name: Optional[str] = None
    ) -> CapabilityResult:
        """
        Execute a single capability with optional fallback.
        
        Args:
            name: Name of the capability to execute
            data: Input data for the capability
            fallback_name: Optional fallback capability name
            
        Returns:
            CapabilityResult
        """
        capability = self.registry.get(name)
        
        if not capability:
            if fallback_name:
                fallback_capability = self.registry.get(fallback_name)
                if fallback_capability:
                    logger.warning(f"Capability '{name}' not found, using fallback '{fallback_name}'")
                    result = await fallback_capability.run(data)
                    result.fallback_used = True
                    return result
            
            return CapabilityResult(
                success=False,
                error=f"Capability '{name}' not found and no fallback available",
                capability_name=name
            )
        
        result = await capability.run_with_retry(data)
        
        # Try fallback if primary failed and fallback is available
        if not result.success and fallback_name:
            fallback_capability = self.registry.get(fallback_name)
            if fallback_capability:
                logger.warning(f"Primary capability '{name}' failed, trying fallback '{fallback_name}'")
                fallback_result = await fallback_capability.run(data)
                if fallback_result.success:
                    fallback_result.fallback_used = True
                    fallback_result.warnings.append(f"Primary capability '{name}' failed: {result.error}")
                    return fallback_result
        
        return result
    
    async def batch_execute(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[CapabilityResult]:
        """
        Execute multiple capability requests in parallel.
        
        Args:
            requests: List of dicts with 'capability', 'data', and optional 'fallback' keys
            
        Returns:
            List of CapabilityResults in the same order as requests
        """
        tasks = []
        
        for request in requests:
            capability_name = request['capability']
            data = request['data']
            fallback_name = request.get('fallback')
            
            task = self.execute_capability(capability_name, data, fallback_name)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_available_pipelines(self) -> List[str]:
        """Get list of available pipeline names"""
        return list(self._pipelines.keys())
    
    def get_pipeline_config(self, name: str) -> Optional[PipelineConfig]:
        """Get pipeline configuration by name"""
        pipeline = self._pipelines.get(name)
        return pipeline.config if pipeline else None