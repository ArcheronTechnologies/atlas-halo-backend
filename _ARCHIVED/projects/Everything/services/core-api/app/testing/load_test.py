"""
Load Testing and Performance Benchmarking System

This module provides comprehensive load testing capabilities to validate
system performance and scalability under various load conditions.
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import random
import uuid
import concurrent.futures
import threading

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load tests"""
    name: str
    base_url: str
    duration_seconds: int = 300  # 5 minutes
    ramp_up_seconds: int = 60    # 1 minute
    concurrent_users: int = 10
    requests_per_second: float = 10.0
    scenarios: List[Dict[str, Any]] = None
    auth_headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.scenarios is None:
            self.scenarios = []
        if self.auth_headers is None:
            self.auth_headers = {}


@dataclass
class RequestResult:
    """Result of a single HTTP request"""
    url: str
    method: str
    status_code: int
    response_time: float
    response_size: int
    error: Optional[str] = None
    timestamp: datetime = None
    user_id: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class LoadTestResults:
    """Comprehensive load test results"""
    config: LoadTestConfig
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    errors_by_type: Dict[str, int]
    response_time_histogram: Dict[str, int]
    status_code_distribution: Dict[int, int]
    throughput_over_time: List[Dict[str, Any]]
    
    @property
    def success_rate(self) -> float:
        return (self.successful_requests / max(self.total_requests, 1)) * 100
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


class LoadTestScenario:
    """Defines a load test scenario with specific request patterns"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.requests: List[Dict[str, Any]] = []
        self.setup_data: Dict[str, Any] = {}
    
    def add_request(self, 
                   method: str, 
                   path: str, 
                   headers: Dict[str, str] = None,
                   data: Any = None,
                   expected_status: int = 200):
        """Add a request to the scenario"""
        self.requests.append({
            'method': method,
            'path': path,
            'headers': headers or {},
            'data': data,
            'expected_status': expected_status
        })
    
    def set_setup_data(self, data: Dict[str, Any]):
        """Set data that will be available during scenario execution"""
        self.setup_data = data
    
    async def execute(self, session: aiohttp.ClientSession, base_url: str, user_id: str) -> List[RequestResult]:
        """Execute the scenario and return results"""
        results = []
        
        for request_spec in self.requests:
            start_time = time.time()
            url = f"{base_url.rstrip('/')}{request_spec['path']}"
            
            try:
                async with session.request(
                    method=request_spec['method'],
                    url=url,
                    headers=request_spec['headers'],
                    json=request_spec['data'] if request_spec['method'] in ['POST', 'PUT', 'PATCH'] else None
                ) as response:
                    response_time = time.time() - start_time
                    response_text = await response.text()
                    
                    result = RequestResult(
                        url=url,
                        method=request_spec['method'],
                        status_code=response.status,
                        response_time=response_time,
                        response_size=len(response_text),
                        user_id=user_id
                    )
                    
                    # Check if status matches expected
                    if response.status != request_spec['expected_status']:
                        result.error = f"Expected status {request_spec['expected_status']}, got {response.status}"
                    
                    results.append(result)
            
            except Exception as e:
                response_time = time.time() - start_time
                result = RequestResult(
                    url=url,
                    method=request_spec['method'],
                    status_code=0,
                    response_time=response_time,
                    response_size=0,
                    error=str(e),
                    user_id=user_id
                )
                results.append(result)
        
        return results


class LoadTester:
    """Main load testing orchestrator"""
    
    def __init__(self):
        self.scenarios: Dict[str, LoadTestScenario] = {}
        self.results_queue = asyncio.Queue()
        self.active_users = 0
        self.results_history: List[RequestResult] = []
    
    def add_scenario(self, scenario: LoadTestScenario):
        """Add a test scenario"""
        self.scenarios[scenario.name] = scenario
    
    def create_api_test_scenarios(self, auth_headers: Dict[str, str] = None):
        """Create standard API test scenarios"""
        
        # Authentication scenario
        auth_scenario = LoadTestScenario("authentication", weight=0.1)
        auth_scenario.add_request("POST", "/v1/auth/login", 
                                data={"email": "test@example.com", "password": "password"})
        self.add_scenario(auth_scenario)
        
        # Component browsing scenario
        browse_scenario = LoadTestScenario("component_browsing", weight=0.4)
        browse_scenario.add_request("GET", "/v1/components", headers=auth_headers)
        browse_scenario.add_request("GET", "/v1/components?page=1&limit=20", headers=auth_headers)
        browse_scenario.add_request("GET", "/v1/components/search?q=resistor", headers=auth_headers)
        self.add_scenario(browse_scenario)
        
        # RFQ creation scenario
        rfq_scenario = LoadTestScenario("rfq_operations", weight=0.3)
        rfq_scenario.add_request("GET", "/v1/companies", headers=auth_headers)
        rfq_scenario.add_request("POST", "/v1/rfqs", headers=auth_headers,
                               data={
                                   "customerId": "test-customer",
                                   "items": [
                                       {"customerPartNumber": "RES-001", "quantity": 100}
                                   ]
                               })
        self.add_scenario(rfq_scenario)
        
        # Intelligence queries scenario  
        intelligence_scenario = LoadTestScenario("intelligence_queries", weight=0.2)
        intelligence_scenario.add_request("GET", "/v1/intelligence/market-analysis/component-123", headers=auth_headers)
        intelligence_scenario.add_request("GET", "/v1/intelligence/supplier-analysis/supplier-456", headers=auth_headers)
        self.add_scenario(intelligence_scenario)
    
    async def run_load_test(self, config: LoadTestConfig) -> LoadTestResults:
        """Execute a complete load test"""
        logger.info(f"Starting load test: {config.name}")
        
        start_time = datetime.now(timezone.utc)
        self.results_history = []
        
        # Create scenarios if none provided
        if not config.scenarios and not self.scenarios:
            self.create_api_test_scenarios(config.auth_headers)
        
        # Start result collector
        collector_task = asyncio.create_task(self._collect_results())
        
        # Start user simulation
        user_tasks = []
        for user_id in range(config.concurrent_users):
            task = asyncio.create_task(
                self._simulate_user(config, f"user-{user_id}", start_time)
            )
            user_tasks.append(task)
        
        # Wait for test duration
        await asyncio.sleep(config.duration_seconds)
        
        # Stop user tasks
        for task in user_tasks:
            task.cancel()
        
        # Wait a bit more for cleanup
        await asyncio.sleep(5)
        
        # Stop collector
        collector_task.cancel()
        
        end_time = datetime.now(timezone.utc)
        
        # Generate results
        results = self._analyze_results(config, start_time, end_time)
        
        logger.info(f"Load test completed: {config.name}")
        logger.info(f"Success rate: {results.success_rate:.1f}%")
        logger.info(f"Avg response time: {results.avg_response_time:.3f}s")
        logger.info(f"Requests per second: {results.requests_per_second:.1f}")
        
        return results
    
    async def _simulate_user(self, config: LoadTestConfig, user_id: str, start_time: datetime):
        """Simulate a single user's behavior"""
        try:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                self.active_users += 1
                
                # Ramp-up delay
                if config.ramp_up_seconds > 0:
                    user_index = int(user_id.split("-")[1])
                    ramp_delay = (config.ramp_up_seconds / config.concurrent_users) * user_index
                    await asyncio.sleep(ramp_delay)
                
                # Calculate request interval
                request_interval = 1.0 / (config.requests_per_second / config.concurrent_users)
                
                while True:
                    # Select scenario based on weights
                    scenario = self._select_scenario()
                    if scenario:
                        try:
                            results = await scenario.execute(session, config.base_url, user_id)
                            
                            # Queue results
                            for result in results:
                                await self.results_queue.put(result)
                            
                        except Exception as e:
                            logger.error(f"User {user_id} scenario error: {e}")
                    
                    # Wait before next request
                    await asyncio.sleep(request_interval)
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"User {user_id} simulation error: {e}")
        finally:
            self.active_users -= 1
    
    def _select_scenario(self) -> Optional[LoadTestScenario]:
        """Select a scenario based on weights"""
        if not self.scenarios:
            return None
        
        # Calculate total weight
        total_weight = sum(scenario.weight for scenario in self.scenarios.values())
        
        # Random selection based on weights
        r = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for scenario in self.scenarios.values():
            cumulative_weight += scenario.weight
            if r <= cumulative_weight:
                return scenario
        
        # Fallback
        return list(self.scenarios.values())[0]
    
    async def _collect_results(self):
        """Collect results from the queue"""
        try:
            while True:
                result = await asyncio.wait_for(self.results_queue.get(), timeout=1.0)
                self.results_history.append(result)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    
    def _analyze_results(self, config: LoadTestConfig, start_time: datetime, end_time: datetime) -> LoadTestResults:
        """Analyze test results and generate report"""
        
        if not self.results_history:
            # Return empty results if no data
            return LoadTestResults(
                config=config,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0.0,
                p50_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                min_response_time=0.0,
                max_response_time=0.0,
                requests_per_second=0.0,
                errors_by_type={},
                response_time_histogram={},
                status_code_distribution={},
                throughput_over_time=[]
            )
        
        # Basic metrics
        total_requests = len(self.results_history)
        successful_requests = sum(1 for r in self.results_history if r.error is None and 200 <= r.status_code < 400)
        failed_requests = total_requests - successful_requests
        
        # Response times
        response_times = [r.response_time for r in self.results_history]
        response_times.sort()
        
        avg_response_time = statistics.mean(response_times)
        p50_response_time = response_times[int(len(response_times) * 0.5)] if response_times else 0
        p95_response_time = response_times[int(len(response_times) * 0.95)] if response_times else 0
        p99_response_time = response_times[int(len(response_times) * 0.99)] if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Throughput
        duration = (end_time - start_time).total_seconds()
        requests_per_second = total_requests / max(duration, 1)
        
        # Error analysis
        errors_by_type = defaultdict(int)
        for result in self.results_history:
            if result.error:
                error_type = type(Exception(result.error)).__name__
                errors_by_type[error_type] += 1
        
        # Status code distribution
        status_code_distribution = defaultdict(int)
        for result in self.results_history:
            status_code_distribution[result.status_code] += 1
        
        # Response time histogram
        response_time_histogram = defaultdict(int)
        for rt in response_times:
            bucket = f"{int(rt * 1000)}ms"  # Convert to ms and bucket
            response_time_histogram[bucket] += 1
        
        # Throughput over time (5-second intervals)
        throughput_over_time = []
        interval = timedelta(seconds=5)
        current_time = start_time
        
        while current_time < end_time:
            next_time = current_time + interval
            interval_requests = [
                r for r in self.results_history 
                if current_time <= r.timestamp < next_time
            ]
            
            throughput_over_time.append({
                'timestamp': current_time.isoformat(),
                'requests': len(interval_requests),
                'rps': len(interval_requests) / interval.total_seconds(),
                'avg_response_time': statistics.mean([r.response_time for r in interval_requests]) if interval_requests else 0,
                'errors': sum(1 for r in interval_requests if r.error)
            })
            
            current_time = next_time
        
        return LoadTestResults(
            config=config,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            errors_by_type=dict(errors_by_type),
            response_time_histogram=dict(response_time_histogram),
            status_code_distribution=dict(status_code_distribution),
            throughput_over_time=throughput_over_time
        )


class PerformanceBenchmark:
    """Performance benchmarking utilities"""
    
    @staticmethod
    async def benchmark_database_operations(num_operations: int = 1000) -> Dict[str, Any]:
        """Benchmark database operations"""
        from ..db.session import get_db_session
        from sqlalchemy import text
        
        results = {
            'operations': num_operations,
            'connection_time': [],
            'query_time': [],
            'transaction_time': []
        }
        
        # Benchmark connections
        for _ in range(min(num_operations, 100)):
            start_time = time.time()
            try:
                with get_db_session() as session:
                    pass
                connection_time = time.time() - start_time
                results['connection_time'].append(connection_time)
            except Exception as e:
                logger.error(f"Database connection benchmark error: {e}")
        
        # Benchmark simple queries
        for _ in range(min(num_operations, 100)):
            start_time = time.time()
            try:
                with get_db_session() as session:
                    session.execute(text("SELECT 1"))
                query_time = time.time() - start_time
                results['query_time'].append(query_time)
            except Exception as e:
                logger.error(f"Database query benchmark error: {e}")
        
        # Calculate statistics
        for metric in ['connection_time', 'query_time', 'transaction_time']:
            if results[metric]:
                times = results[metric]
                results[f'{metric}_avg'] = statistics.mean(times)
                results[f'{metric}_p95'] = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]
                results[f'{metric}_min'] = min(times)
                results[f'{metric}_max'] = max(times)
        
        return results
    
    @staticmethod
    async def benchmark_cache_operations(num_operations: int = 1000) -> Dict[str, Any]:
        """Benchmark cache operations"""
        from ..cache.advanced_cache import advanced_cache
        
        results = {
            'operations': num_operations,
            'set_time': [],
            'get_time': [],
            'delete_time': []
        }
        
        # Benchmark cache sets
        for i in range(min(num_operations, 100)):
            start_time = time.time()
            try:
                await advanced_cache.set("benchmark", f"key-{i}", f"value-{i}")
                set_time = time.time() - start_time
                results['set_time'].append(set_time)
            except Exception as e:
                logger.error(f"Cache set benchmark error: {e}")
        
        # Benchmark cache gets
        for i in range(min(num_operations, 100)):
            start_time = time.time()
            try:
                await advanced_cache.get("benchmark", f"key-{i}")
                get_time = time.time() - start_time
                results['get_time'].append(get_time)
            except Exception as e:
                logger.error(f"Cache get benchmark error: {e}")
        
        # Calculate statistics
        for metric in ['set_time', 'get_time', 'delete_time']:
            if results[metric]:
                times = results[metric]
                results[f'{metric}_avg'] = statistics.mean(times)
                results[f'{metric}_p95'] = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]
                results[f'{metric}_min'] = min(times)
                results[f'{metric}_max'] = max(times)
        
        return results


# Predefined test configurations
def create_smoke_test_config(base_url: str, auth_headers: Dict[str, str] = None) -> LoadTestConfig:
    """Create a light smoke test configuration"""
    return LoadTestConfig(
        name="smoke_test",
        base_url=base_url,
        duration_seconds=60,
        ramp_up_seconds=10,
        concurrent_users=2,
        requests_per_second=2.0,
        auth_headers=auth_headers or {}
    )


def create_load_test_config(base_url: str, auth_headers: Dict[str, str] = None) -> LoadTestConfig:
    """Create a standard load test configuration"""
    return LoadTestConfig(
        name="load_test",
        base_url=base_url,
        duration_seconds=300,
        ramp_up_seconds=60,
        concurrent_users=10,
        requests_per_second=10.0,
        auth_headers=auth_headers or {}
    )


def create_stress_test_config(base_url: str, auth_headers: Dict[str, str] = None) -> LoadTestConfig:
    """Create a stress test configuration"""
    return LoadTestConfig(
        name="stress_test",
        base_url=base_url,
        duration_seconds=600,
        ramp_up_seconds=120,
        concurrent_users=50,
        requests_per_second=50.0,
        auth_headers=auth_headers or {}
    )


def create_spike_test_config(base_url: str, auth_headers: Dict[str, str] = None) -> LoadTestConfig:
    """Create a spike test configuration"""
    return LoadTestConfig(
        name="spike_test",
        base_url=base_url,
        duration_seconds=180,
        ramp_up_seconds=10,  # Quick ramp up
        concurrent_users=100,
        requests_per_second=100.0,
        auth_headers=auth_headers or {}
    )


# Global load tester instance
load_tester = LoadTester()