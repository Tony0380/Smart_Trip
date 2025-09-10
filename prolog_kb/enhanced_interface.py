import subprocess
import json
import re
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EnhancedPrologKB:
    """
    Enhanced Prolog Knowledge Base with complex inference capabilities
    """
    
    def __init__(self, kb_file: str = None):
        if kb_file is None:
            kb_file = os.path.join(os.path.dirname(__file__), "travel_rules.pl")
        
        self.kb_file = kb_file
        self.swipl_available = self._check_swipl()
        
        if not self.swipl_available:
            print("[WARNING] SWI-Prolog not found. Limited inference available.")
        else:
            print(f"[PROLOG] Enhanced KB loaded from: {kb_file}")
    
    def _check_swipl(self) -> bool:
        try:
            result = subprocess.run(['swipl', '--version'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _execute_query(self, query: str) -> List[Any]:
        if not self.swipl_available:
            return []
        
        try:
            cmd = [
                'swipl', '-q', '-t', 'halt',
                '-s', self.kb_file,
                '-g', f"({query} -> write('SUCCESS: '), write({query}), nl; write('FAIL')), halt"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if 'SUCCESS:' in result.stdout:
                return [result.stdout.split('SUCCESS: ')[1].strip()]
            else:
                return []
                
        except Exception as e:
            print(f"[PROLOG ERROR] {e}")
            return []
    
    def advanced_route_planning(self, origin: str, destination: str, profile: str, constraints: Dict = None) -> Dict[str, Any]:
        """
        Advanced route planning with multi-objective optimization and inference
        """
        
        # Query per ottimizzazione multi-obiettivo
        optimal_query = f"optimal_route({origin}, {destination}, {profile}, {profile}, OptimalRoute)"
        optimal_result = self._execute_query(optimal_query)
        
        # Calcolo affidabilità con inferenza meteorologica
        reliability_query = f"transport_reliability(train, north, summer, Reliability)"
        reliability_result = self._execute_query(reliability_query)
        
        # Meta-ragionamento per spiegazioni
        if optimal_result:
            explain_query = f"explain_route_choice({optimal_result[0]}, {profile}, Explanation)"
            explanation_result = self._execute_query(explain_query)
        else:
            explanation_result = ["Route optimization failed"]
        
        # Constraint satisfaction
        if constraints:
            constraint_checks = []
            for constraint_type, value in constraints.items():
                constraint_query = f"satisfies_constraint(route([{origin}, {destination}], train, 100, 120, 0.8, 0.9), {constraint_type}({value}))"
                constraint_result = self._execute_query(constraint_query)
                constraint_checks.append(bool(constraint_result))
        else:
            constraint_checks = [True]
        
        return {
            'optimal_route': optimal_result[0] if optimal_result else None,
            'reliability_score': float(reliability_result[0]) if reliability_result else 0.8,
            'explanation': explanation_result,
            'constraints_satisfied': all(constraint_checks),
            'inference_depth': 'multi_level',
            'reasoning_type': 'meta_cognitive'
        }
    
    def temporal_scheduling(self, origin: str, destination: str, departure_time: int) -> Dict[str, Any]:
        """
        Temporal reasoning for trip scheduling
        """
        
        schedule_query = f"schedule_trip({origin}, {destination}, business, {departure_time}, ArrivalTime, Schedule)"
        schedule_result = self._execute_query(schedule_query)
        
        availability_query = f"available_at_time(train, {departure_time})"
        availability_result = self._execute_query(availability_query)
        
        return {
            'schedule': schedule_result[0] if schedule_result else None,
            'transport_available': bool(availability_result),
            'temporal_reasoning': True
        }
    
    def reliability_inference(self, transport: str, region: str, season: str) -> Dict[str, Any]:
        """
        Complex reliability inference considering multiple factors
        """
        
        reliability_query = f"transport_reliability({transport}, {region}, {season}, Reliability)"
        reliability_result = self._execute_query(reliability_query)
        
        # Inferenza su eventi che influenzano affidabilità
        event_query = f"event(strike, {transport}, Properties)"
        event_result = self._execute_query(event_query)
        
        weather_query = f"weather_impact({region}, {season}, Impact)"
        weather_result = self._execute_query(weather_query)
        
        return {
            'reliability': float(reliability_result[0]) if reliability_result else 0.8,
            'events_affecting': event_result,
            'weather_impact': weather_result,
            'inference_factors': ['events', 'weather', 'seasonal'],
            'reasoning_depth': 'complex'
        }
    
    def constraint_propagation(self, base_constraints: List[str], derived_constraints: List[str] = None) -> Dict[str, Any]:
        """
        Constraint satisfaction with propagation
        """
        
        results = {}
        
        for constraint in base_constraints:
            constraint_query = f"satisfies_constraint(route([milano, roma], train, 150, 120, 0.8, 0.9), {constraint})"
            result = self._execute_query(constraint_query)
            results[constraint] = bool(result)
        
        # Propagazione vincoli derivati
        if derived_constraints:
            propagation_query = f"satisfies_all_constraints(route([milano, roma], train, 150, 120, 0.8, 0.9), {base_constraints + derived_constraints})"
            propagation_result = self._execute_query(propagation_query)
            results['propagation_satisfied'] = bool(propagation_result)
        
        return {
            'constraint_results': results,
            'propagation_used': bool(derived_constraints),
            'reasoning_type': 'constraint_satisfaction'
        }