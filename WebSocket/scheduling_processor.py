"""
Scheduling algorithm processor for WebSocket service
Reuses the algorithm processing logic from RPC implementation
"""

import os
import sys
import json
import time
import tempfile
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from Environment.dfjspt_env import FjspMaEnv
from Algorithms.DispatchingRules.dfjspt_rule1_EST_EET import est_eet_rule_weighted
from Algorithms.DispatchingRules.dfjspt_rule2_EST_SPT import est_spt_rule_weighted
from Algorithms.GA.ga_scheduler import FjspMaEnv as GA_FjspMaEnv, GAConfig, GAScheduler

# Configure logging
logger = logging.getLogger(__name__)

class WebSocketSchedulingProcessor:
    """Scheduling algorithm processor for WebSocket service"""
    
    def __init__(self):
        self.name = "WebSocketSchedulingProcessor"
        self.supported_algorithms = {
            "est_eet_weighted": "Earliest Start Time - Earliest End Time with Priority Weighting",
            "est_spt_weighted": "Earliest Start Time - Shortest Processing Time with Priority Weighting",
            "ga": "Genetic Algorithm with Priority Weighting"
        }
        logger.info(f"Initialized {self.name}")
    
    def _create_temp_input_file(self, input_data: Dict[str, Any]) -> str:
        """Create temporary input file from JSON data"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(input_data, temp_file, indent=2)
        temp_file.close()
        return temp_file.name
    
    def _cleanup_temp_file(self, file_path: str) -> None:
        """Clean up temporary file"""
        try:
            os.unlink(file_path)
        except OSError:
            pass
    
    def _create_output_path(self, algorithm_name: str, input_filename: str) -> str:
        """Create output file path"""
        output_dir = os.path.join(project_root, 'Data', 'OutputData')
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base name from input filename
        base_name = os.path.splitext(os.path.basename(input_filename))[0]
        output_filename = f"output_{algorithm_name}_{base_name}.json"
        return os.path.join(output_dir, output_filename)
    
    def est_eet_weighted_scheduling(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """EST-EET weighted scheduling algorithm"""
        try:
            start_time = time.time()
            
            # Get parameters
            alpha = data.get("alpha", 0.7)
            input_data = data.get("input_data", {})
            input_filename = data.get("input_filename", "input.json")
            
            logger.info(f"Running EST-EET weighted scheduling with alpha={alpha}")
            
            # # Create temporary input file
            # temp_input_file = self._create_temp_input_file(input_data)
            
            # Create environment
            env = FjspMaEnv({
                'inputdata_json': input_data
            })
            
            # Run scheduling algorithm
            makespan, total_reward = est_eet_rule_weighted(env, alpha=alpha, verbose=False)
            
            # Generate output file
            output_path = self._create_output_path("est_eet_weighted", input_filename)
            env.build_and_save_output_json(output_path)
            
            # Read output data
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            
            execution_time = time.time() - start_time
            
            # # Cleanup
            # self._cleanup_temp_file(temp_input_file)
            
            logger.info(f"EST-EET weighted scheduling completed in {execution_time:.2f}s, makespan: {makespan}")
            
            return {
                "algorithm": "est_eet_weighted",
                "status": "success",
                "makespan": makespan,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "output_file": output_path,
                "output_data": output_data
            }
            
        except Exception as e:
            logger.error(f"EST-EET weighted scheduling error: {e}")
            return {
                "algorithm": "est_eet_weighted",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def est_spt_weighted_scheduling(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """EST-SPT weighted scheduling algorithm"""
        try:
            start_time = time.time()
            alpha = data.get("alpha", 0.7)
            input_data = data.get("input_data", {})
            input_filename = data.get("input_filename", "input.json")
            logger.info(f"Running EST-SPT weighted scheduling with alpha={alpha}")
            # temp_input_file = self._create_temp_input_file(input_data)
            env = FjspMaEnv({
                'inputdata_json': input_data
            })
            makespan, total_reward = est_spt_rule_weighted(env, alpha=alpha, verbose=False)
            output_path = self._create_output_path("est_spt_weighted", input_filename)
            env.build_and_save_output_json(output_path)
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            execution_time = time.time() - start_time
            # self._cleanup_temp_file(temp_input_file)
            logger.info(f"EST-SPT weighted scheduling completed in {execution_time:.2f}s, makespan: {makespan}")
            return {
                "algorithm": "est_spt_weighted",
                "status": "success",
                "makespan": makespan,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "output_file": output_path,
                "output_data": output_data
            }
        except Exception as e:
            logger.error(f"EST-SPT weighted scheduling error: {e}")
            return {
                "algorithm": "est_spt_weighted",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def ga_scheduling(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Genetic Algorithm scheduling"""
        try:
            start_time = time.time()
            alpha = data.get("alpha", 0.7)
            input_data = data.get("input_data", {})
            input_filename = data.get("input_filename", "input.json")
            pop = data.get("pop", 100)
            gen = data.get("gen", 30)
            seed = data.get("seed", None)
            logger.info(f"Running GA scheduling with alpha={alpha}, pop={pop}, gen={gen}, seed={seed}")
            # temp_input_file = self._create_temp_input_file(input_data)
            env = GA_FjspMaEnv({
                'inputdata_json': input_data
            })
            config = GAConfig(population_size=pop, generations=gen, alpha=alpha, seed=seed)
            ga = GAScheduler(env, config)
            best = ga.evolve()
            makespan = best.schedule_result.final_makespan
            output_path = self._create_output_path("ga", input_filename)
            best.schedule_result.build_and_save_output_json(output_path)
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            execution_time = time.time() - start_time
            # self._cleanup_temp_file(temp_input_file)
            logger.info(f"GA scheduling completed in {execution_time:.2f}s, makespan: {makespan}")
            return {
                "algorithm": "ga",
                "status": "success",
                "makespan": makespan,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "output_file": output_path,
                "output_data": output_data
            }
        except Exception as e:
            logger.error(f"GA scheduling error: {e}")
            return {
                "algorithm": "ga",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_supported_algorithms(self) -> Dict[str, Any]:
        """Get list of supported algorithms"""
        return {
            "status": "success",
            "algorithms": self.supported_algorithms,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_algorithm_info(self, algorithm_name: str) -> Dict[str, Any]:
        """Get information about a specific algorithm"""
        if algorithm_name in self.supported_algorithms:
            return {
                "status": "success",
                "algorithm": algorithm_name,
                "description": self.supported_algorithms[algorithm_name],
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "error",
                "error": f"Algorithm '{algorithm_name}' not found",
                "available_algorithms": list(self.supported_algorithms.keys()),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_scheduling_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process scheduling request based on algorithm type"""
        algorithm = request_data.get("algorithm", "est_eet_weighted")
        if algorithm == "est_eet_weighted":
            return self.est_eet_weighted_scheduling(request_data)
        elif algorithm == "est_spt_weighted":
            return self.est_spt_weighted_scheduling(request_data)
        elif algorithm == "ga":
            return self.ga_scheduling(request_data)
        else:
            return {
                "status": "error",
                "error": f"Unsupported algorithm: {algorithm}",
                "supported_algorithms": list(self.supported_algorithms.keys()),
                "timestamp": datetime.now().isoformat()
            }

# Create global instance
scheduling_processor = WebSocketSchedulingProcessor() 