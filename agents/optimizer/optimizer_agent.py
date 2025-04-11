"""
Optimizer Agent - Handles parameter optimization for trading strategies.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms
import random

from agents.base_agent import BaseAgent
from services.market_data import MarketDataService
from utils.performance_metrics import calculate_sharpe_ratio, calculate_max_drawdown

class OptimizerAgent(BaseAgent):
    """
    Optimizer Agent that handles parameter optimization for trading strategies.
    
    This agent:
    - Runs parameter optimization using genetic algorithms
    - Performs grid search for parameter combinations
    - Evaluates strategy performance
    - Provides optimal parameter recommendations
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 config: Dict[str, Any], 
                 parent_id: Optional[str] = None):
        """
        Initialize the optimizer agent.
        
        Args:
            name: Human-readable name for the agent
            description: Description of the agent's purpose
            config: Configuration dictionary for the agent
            parent_id: Optional ID of parent agent
        """
        super().__init__(name, description, config, parent_id, agent_type="optimizer")
        
        # Optimization configuration
        self.optimization_method = config.get("optimization_method", "genetic")
        self.population_size = config.get("population_size", 50)
        self.generations = config.get("generations", 20)
        self.mutation_rate = config.get("mutation_rate", 0.2)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        
        # Grid search configuration
        self.grid_search_steps = config.get("grid_search_steps", 5)
        
        # Performance metrics
        self.metrics = config.get("metrics", ["sharpe_ratio", "max_drawdown", "profit_factor"])
        
        # Market data service
        self.market_data_service = MarketDataService(
            api_key=config.get("api_key"),
            secret=config.get("secret")
        )
        
        # Initialize DEAP framework
        self._setup_deap()
        
        self.logger.info(f"Optimizer Agent initialized with {self.optimization_method} optimization method")
    
    def _setup_deap(self) -> None:
        """Setup DEAP framework for genetic algorithm optimization."""
        # Create fitness class
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # Create individual class
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def optimize_strategy(self, 
                         strategy: Any,
                         parameters: Dict[str, Tuple[float, float]],
                         data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using specified method.
        
        Args:
            strategy: Strategy to optimize
            parameters: Dictionary of parameter names and their ranges
            data: Market data for optimization
            
        Returns:
            Dictionary containing optimization results
        """
        if self.optimization_method == "genetic":
            return self._optimize_genetic(strategy, parameters, data)
        elif self.optimization_method == "grid":
            return self._optimize_grid(strategy, parameters, data)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
    
    def _optimize_genetic(self,
                         strategy: Any,
                         parameters: Dict[str, Tuple[float, float]],
                         data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using genetic algorithm.
        
        Args:
            strategy: Strategy to optimize
            parameters: Dictionary of parameter names and their ranges
            data: Market data for optimization
            
        Returns:
            Dictionary containing optimization results
        """
        # Register individual creation
        self.toolbox.register("attr_float", random.uniform, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=len(parameters))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register evaluation function
        self.toolbox.register("evaluate", self._evaluate_strategy, strategy=strategy,
                            parameters=parameters, data=data)
        
        # Create initial population
        pop = self.toolbox.population(n=self.population_size)
        
        # Run genetic algorithm
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=self.crossover_rate,
                                     mutpb=self.mutation_rate, ngen=self.generations,
                                     stats=stats, halloffame=hof, verbose=True)
        
        # Get best individual
        best_individual = hof[0]
        best_parameters = self._decode_parameters(best_individual, parameters)
        
        # Evaluate best parameters
        best_performance = self._evaluate_strategy(best_individual, strategy, parameters, data)
        
        return {
            "method": "genetic",
            "best_parameters": best_parameters,
            "best_performance": best_performance,
            "optimization_log": log,
            "generations": self.generations,
            "population_size": self.population_size
        }
    
    def _optimize_grid(self,
                      strategy: Any,
                      parameters: Dict[str, Tuple[float, float]],
                      data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy: Strategy to optimize
            parameters: Dictionary of parameter names and their ranges
            data: Market data for optimization
            
        Returns:
            Dictionary containing optimization results
        """
        best_performance = float('-inf')
        best_parameters = None
        results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_grid_combinations(parameters)
        
        # Evaluate each combination
        for params in param_combinations:
            performance = self._evaluate_strategy(params, strategy, parameters, data)
            results.append({
                "parameters": self._decode_parameters(params, parameters),
                "performance": performance
            })
            
            if performance > best_performance:
                best_performance = performance
                best_parameters = self._decode_parameters(params, parameters)
        
        return {
            "method": "grid",
            "best_parameters": best_parameters,
            "best_performance": best_performance,
            "all_results": results,
            "total_combinations": len(param_combinations)
        }
    
    def _evaluate_strategy(self,
                          individual: List[float],
                          strategy: Any,
                          parameters: Dict[str, Tuple[float, float]],
                          data: Dict[str, Any]) -> float:
        """
        Evaluate strategy performance with given parameters.
        
        Args:
            individual: Encoded parameter values
            strategy: Strategy to evaluate
            parameters: Dictionary of parameter names and their ranges
            data: Market data for evaluation
            
        Returns:
            Fitness score (higher is better)
        """
        # Decode parameters
        decoded_params = self._decode_parameters(individual, parameters)
        
        # Update strategy parameters
        for param_name, value in decoded_params.items():
            setattr(strategy, param_name, value)
        
        # Run strategy on data
        results = strategy.run(data)
        
        # Calculate performance metrics
        returns = results.get("returns", [])
        if not returns:
            return float('-inf')
        
        # Calculate fitness based on configured metrics
        fitness = 0
        for metric in self.metrics:
            if metric == "sharpe_ratio":
                fitness += calculate_sharpe_ratio(returns)
            elif metric == "max_drawdown":
                fitness -= calculate_max_drawdown(returns)
            elif metric == "profit_factor":
                profits = [r for r in returns if r > 0]
                losses = [r for r in returns if r < 0]
                if sum(losses) != 0:
                    fitness += sum(profits) / abs(sum(losses))
        
        return fitness
    
    def _decode_parameters(self,
                          individual: List[float],
                          parameters: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Decode normalized parameter values to actual ranges.
        
        Args:
            individual: Encoded parameter values
            parameters: Dictionary of parameter names and their ranges
            
        Returns:
            Dictionary of decoded parameter values
        """
        decoded = {}
        for i, (param_name, (min_val, max_val)) in enumerate(parameters.items()):
            # Scale normalized value to parameter range
            decoded[param_name] = min_val + (max_val - min_val) * individual[i]
        return decoded
    
    def _generate_grid_combinations(self,
                                  parameters: Dict[str, Tuple[float, float]]) -> List[List[float]]:
        """
        Generate parameter combinations for grid search.
        
        Args:
            parameters: Dictionary of parameter names and their ranges
            
        Returns:
            List of parameter combinations
        """
        # Generate parameter values for each parameter
        param_values = {}
        for param_name, (min_val, max_val) in parameters.items():
            param_values[param_name] = np.linspace(min_val, max_val, self.grid_search_steps)
        
        # Generate all combinations
        from itertools import product
        combinations = []
        for values in product(*param_values.values()):
            combinations.append(list(values))
        
        return combinations
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """
        Get the latest optimization results.
        
        Returns:
            Dictionary containing optimization results
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "optimization_method": self.optimization_method,
            "metrics": self.metrics,
            "configuration": {
                "population_size": self.population_size,
                "generations": self.generations,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "grid_search_steps": self.grid_search_steps
            }
        } 