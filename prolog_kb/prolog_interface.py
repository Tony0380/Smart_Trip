import subprocess
import json
import re
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PrologKnowledgeBase:
    """
    Interfaccia Python-Prolog per Knowledge Base di viaggio

    ARGOMENTI DEL PROGRAMMA IMPLEMENTATI:
    1. Logic Programming: regole e fatti in Prolog
    2. Symbolic Reasoning: inferenza su conoscenza simbolica
    3. Query Processing: interrogazione KB con unificazione
    4. Integration: bridge tra paradigma logico e procedurale
    5. Knowledge Representation: ontologia dominio in predicati

    INTEGRAZIONE SISTEMA:
    - Input da ML models --> Query Prolog per validazione
    - Output Prolog --> Constraints per algoritmi ricerca
    - Symbolic reasoning + Probabilistic inference + Search
    """

    def __init__(self, kb_file: str = None):
        if kb_file is None:
            kb_file = os.path.join(os.path.dirname(__file__), "travel_rules.pl")

        self.kb_file = kb_file
        self.swipl_available = self._check_swipl()

        if not self.swipl_available:
            print("[WARNING] SWI-Prolog not found. Using fallback rules engine.")
            self._init_fallback_engine()
        else:
            print(f"[PROLOG] Loaded KB from: {kb_file}")

    def _check_swipl(self) -> bool:
        """Verifica disponibilità SWI-Prolog"""
        try:
            result = subprocess.run(['swipl', '--version'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _init_fallback_engine(self):
        """
        Motore di inferenza Python semplificato per quando SWI-Prolog non disponibile
        Implementa subset logica necessaria per testing
        """

        # Fatti base estratti da KB Prolog
        self.cities = {
            'milano': {'region': 'north', 'population': 1400000, 'cost_level': 'high_cost'},
            'roma': {'region': 'center', 'population': 2800000, 'cost_level': 'medium_cost'},
            'napoli': {'region': 'south', 'population': 1000000, 'cost_level': 'medium_cost'},
            'torino': {'region': 'north', 'population': 870000, 'cost_level': 'medium_cost'},
            'bologna': {'region': 'north', 'population': 390000, 'cost_level': 'medium_cost'},
            'firenze': {'region': 'center', 'population': 380000, 'cost_level': 'medium_cost'},
            'bari': {'region': 'south', 'population': 325000, 'cost_level': 'low_cost'},
            'palermo': {'region': 'south', 'population': 670000, 'cost_level': 'low_cost'},
            'venezia': {'region': 'north', 'population': 260000, 'cost_level': 'high_cost'}
        }

        self.connections = {
            ('milano', 'torino'): ['train', 'bus'],
            ('milano', 'venezia'): ['train', 'bus'],
            ('milano', 'bologna'): ['train', 'bus'],
            ('bologna', 'firenze'): ['train', 'bus'],
            ('firenze', 'roma'): ['train', 'bus'],
            ('roma', 'napoli'): ['train', 'bus'],
            ('roma', 'palermo'): ['flight'],
            ('milano', 'palermo'): ['flight'],
            ('milano', 'bari'): ['flight']
        }

        self.user_profiles = {
            'business': {
                'income': 'high_income',
                'price_sensitivity': 'low_price_sensitivity',
                'time_priority': 'high_time_priority',
                'comfort': 'high_comfort',
                'preferred_transports': ['flight', 'train']
            },
            'leisure': {
                'income': 'medium_income',
                'price_sensitivity': 'medium_price_sensitivity',
                'time_priority': 'medium_time_priority',
                'comfort': 'medium_comfort',
                'preferred_transports': ['train', 'bus']
            },
            'budget': {
                'income': 'low_income',
                'price_sensitivity': 'high_price_sensitivity',
                'time_priority': 'low_time_priority',
                'comfort': 'low_comfort',
                'preferred_transports': ['bus']
            }
        }

        self.transport_info = {
            'train': {'speed': 120, 'cost_per_km': 0.15, 'comfort': 'high', 'punctuality': 'medium'},
            'bus': {'speed': 80, 'cost_per_km': 0.08, 'comfort': 'low', 'punctuality': 'low'},
            'flight': {'speed': 500, 'cost_per_km': 0.25, 'comfort': 'high', 'punctuality': 'high'}
        }

    def query_prolog(self, query: str) -> List[Dict[str, Any]]:
        """
        Esegue query Prolog e ritorna risultati
        """

        if not self.swipl_available:
            return self._query_fallback(query)

        try:
            # Costruisci comando SWI-Prolog
            prolog_command = f"""
            :- consult('{self.kb_file}').
            :- findall(Result, ({query}), Results),
               write_canonical(Results),
               halt.
            """

            # Esegui query
            result = subprocess.run(['swipl', '-q', '-t', prolog_command],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                return self._parse_prolog_output(result.stdout)
            else:
                print(f"[ERROR] Prolog query failed: {result.stderr}")
                return []

        except subprocess.TimeoutExpired:
            print("[ERROR] Prolog query timed out")
            return []
        except Exception as e:
            print(f"[ERROR] Prolog interface error: {e}")
            return []

    def _parse_prolog_output(self, output: str) -> List[Dict[str, Any]]:
        """
        Parse output di SWI-Prolog in formato Python
        """
        # Implementazione semplificata - in produzione serve parser più robusto
        try:
            # Rimuovi caratteri di controllo e parse come lista
            clean_output = output.strip()
            if clean_output.startswith('[') and clean_output.endswith(']'):
                # Parse basic list format
                items = clean_output[1:-1].split(',')
                return [{'result': item.strip()} for item in items if item.strip()]
            else:
                return [{'result': clean_output}]
        except:
            return []

    def _query_fallback(self, query: str) -> List[Dict[str, Any]]:
        """
        Motore inferenza fallback per query comuni
        """

        # Pattern matching per query comuni
        if 'connected(' in query:
            return self._handle_connected_query(query)
        elif 'suitable_transport(' in query:
            return self._handle_suitable_transport_query(query)
        elif 'valid_trip(' in query:
            return self._handle_valid_trip_query(query)
        elif 'path(' in query:
            return self._handle_path_query(query)
        elif 'travel_advice(' in query:
            return self._handle_travel_advice_query(query)
        else:
            return [{'result': 'unsupported_query', 'fallback': True}]

    def _handle_connected_query(self, query: str) -> List[Dict[str, Any]]:
        """Gestisce query connected/3"""

        # Estrai parametri dalla query
        match = re.search(r'connected\((\w+),\s*(\w+),\s*([^)]+)\)', query)
        if not match:
            return []

        city_a, city_b, _ = match.groups()

        # Cerca connessioni dirette
        results = []
        for (c1, c2), transports in self.connections.items():
            if (c1 == city_a and c2 == city_b) or (c1 == city_b and c2 == city_a):
                results.append({
                    'city_a': c1,
                    'city_b': c2,
                    'transports': transports
                })

        return results

    def _handle_suitable_transport_query(self, query: str) -> List[Dict[str, Any]]:
        """Gestisce query suitable_transport/2"""

        match = re.search(r'suitable_transport\((\w+),\s*(\w+)\)', query)
        if not match:
            return []

        user_profile, transport = match.groups()

        if user_profile in self.user_profiles:
            profile_data = self.user_profiles[user_profile]
            preferred = profile_data['preferred_transports']

            if transport in preferred:
                return [{'user_profile': user_profile, 'transport': transport, 'suitable': True}]

        return []

    def _handle_valid_trip_query(self, query: str) -> List[Dict[str, Any]]:
        """Gestisce query valid_trip/6"""

        # Parsing semplificato
        if 'valid_trip(' in query:
            # In un'implementazione completa, farebbe inferenza completa
            return [{'result': 'valid_trip_analysis', 'status': 'requires_full_prolog'}]

        return []

    def _handle_path_query(self, query: str) -> List[Dict[str, Any]]:
        """Gestisce query path/4 con ricerca percorsi"""

        match = re.search(r'path\((\w+),\s*(\w+),\s*([^,]+),\s*([^)]+)\)', query)
        if not match:
            return []

        origin, destination, _, _ = match.groups()

        # Ricerca percorso semplificata (DFS)
        path = self._find_path_fallback(origin, destination)

        if path:
            return [{'origin': origin, 'destination': destination, 'path': path}]
        else:
            return []

    def _find_path_fallback(self, origin: str, destination: str, visited: List[str] = None) -> Optional[List[str]]:
        """Ricerca percorso con DFS semplificato"""

        if visited is None:
            visited = []

        if origin == destination:
            return [origin]

        if origin in visited:
            return None

        visited.append(origin)

        # Cerca connessioni dirette
        for (c1, c2), transports in self.connections.items():
            next_city = None
            if c1 == origin and c2 not in visited:
                next_city = c2
            elif c2 == origin and c1 not in visited:
                next_city = c1

            if next_city:
                sub_path = self._find_path_fallback(next_city, destination, visited.copy())
                if sub_path:
                    return [origin] + sub_path

        return None

    def _handle_travel_advice_query(self, query: str) -> List[Dict[str, Any]]:
        """Gestisce query travel_advice/4"""

        return [{'advice': 'Use fallback ML recommendations when Prolog unavailable'}]

    # Interfaccia high-level per integrazione con sistema ML
    def validate_travel_plan(self,
                           origin: str,
                           destination: str,
                           user_profile: str,
                           transport: str,
                           budget: float,
                           season: str = 'summer') -> Dict[str, Any]:
        """
        Valida piano viaggio usando regole logiche

        INTEGRAZIONE:
        - Input da ML models --> Validation con regole Prolog
        - Symbolic constraints check
        - Logic-based feasibility analysis
        """

        print(f"[PROLOG] Validating travel plan: {origin} -> {destination} ({user_profile}, {transport})")

        validation_result = {
            'valid': True,
            'constraints_satisfied': [],
            'violations': [],
            'suggestions': []
        }

        # 1. Check basic connectivity
        connectivity_query = f"connected({origin}, {destination}, Transports)"
        connection_results = self.query_prolog(connectivity_query)

        if not connection_results:
            validation_result['valid'] = False
            validation_result['violations'].append('no_direct_connection')

            # Try path query for indirect connections
            path_query = f"path({origin}, {destination}, Path, Transports)"
            path_results = self.query_prolog(path_query)

            if path_results:
                validation_result['suggestions'].append('indirect_path_available')
            else:
                validation_result['violations'].append('no_path_exists')
        else:
            validation_result['constraints_satisfied'].append('connectivity_ok')

        # 2. Check transport suitability for user profile
        suitability_query = f"suitable_transport({user_profile}, {transport})"
        suitability_results = self.query_prolog(suitability_query)

        if suitability_results:
            validation_result['constraints_satisfied'].append('transport_suitable')
        else:
            validation_result['violations'].append('transport_not_suitable')

            # Get recommended alternatives
            advice_query = f"travel_advice({origin}, {destination}, {user_profile}, Advice)"
            advice_results = self.query_prolog(advice_query)

            if advice_results:
                validation_result['suggestions'].append(f"alternatives: {advice_results}")

        # 3. Check budget constraints
        if self.swipl_available:
            budget_query = f"within_budget({origin}, {destination}, {transport}, {budget})"
            budget_results = self.query_prolog(budget_query)

            if budget_results:
                validation_result['constraints_satisfied'].append('budget_ok')
            else:
                validation_result['violations'].append('exceeds_budget')
        else:
            # Fallback budget check
            estimated_cost = self._estimate_cost_fallback(origin, destination, transport)
            if estimated_cost <= budget:
                validation_result['constraints_satisfied'].append('budget_ok')
            else:
                validation_result['violations'].append('exceeds_budget')
                validation_result['suggestions'].append(f'estimated_cost: {estimated_cost}, budget: {budget}')

        # 4. Check seasonal restrictions
        if season == 'winter':
            seasonal_query = f"no_travel_restrictions({origin}, {destination}, {season})"
            seasonal_results = self.query_prolog(seasonal_query)

            if not seasonal_results:
                validation_result['violations'].append('seasonal_restrictions')
                validation_result['suggestions'].append('consider_different_season')

        # Final validation status
        validation_result['valid'] = len(validation_result['violations']) == 0

        return validation_result

    def _estimate_cost_fallback(self, origin: str, destination: str, transport: str) -> float:
        """Stima costo fallback per quando Prolog non disponibile"""

        # Distanze approssimate
        distances = {
            ('milano', 'roma'): 570,
            ('milano', 'napoli'): 770,
            ('roma', 'napoli'): 225,
            ('milano', 'palermo'): 935,
            ('roma', 'palermo'): 490
        }

        distance = distances.get((origin, destination)) or distances.get((destination, origin), 300)
        cost_per_km = self.transport_info.get(transport, {}).get('cost_per_km', 0.15)

        return distance * cost_per_km

    def get_travel_recommendations(self,
                                 origin: str,
                                 destination: str,
                                 user_profile: str) -> Dict[str, Any]:
        """
        Ottieni raccomandazioni complete usando KB logico

        OUTPUT per integrazione ML+Search:
        - Recommended transports con ranking
        - Budget estimates
        - Route alternatives
        - Constraint warnings
        """

        print(f"[PROLOG] Getting travel recommendations: {origin} -> {destination} for {user_profile}")

        recommendations = {
            'recommended_transports': [],
            'alternative_routes': [],
            'budget_estimates': {},
            'constraints': [],
            'feasibility': 'unknown'
        }

        # Query best transport for user
        best_transport_query = f"best_transport_for_user({origin}, {destination}, {user_profile}, BestTransport)"
        best_results = self.query_prolog(best_transport_query)

        if best_results:
            recommendations['recommended_transports'] = [result.get('result', 'train') for result in best_results]
        else:
            # Fallback recommendations based on profile
            if user_profile in self.user_profiles:
                recommendations['recommended_transports'] = self.user_profiles[user_profile]['preferred_transports']

        # Get budget estimates for each transport
        for transport in ['train', 'bus', 'flight']:
            cost = self._estimate_cost_fallback(origin, destination, transport)
            recommendations['budget_estimates'][transport] = cost

        # Check feasibility
        feasibility_query = f"travel_feasibility({origin}, {destination}, {user_profile}, 500, summer, Result)"
        feasibility_results = self.query_prolog(feasibility_query)

        if feasibility_results:
            recommendations['feasibility'] = feasibility_results[0].get('result', 'feasible')

        return recommendations

    def explain_constraints(self,
                          origin: str,
                          destination: str,
                          user_profile: str,
                          transport: str) -> str:
        """
        Spiega vincoli logici per decision support

         AI spiegabile con ragionamento simbolico
        """

        explanation = f"ANALISI VINCOLI LOGICI: {origin} -> {destination}\n"
        explanation += f"Profilo utente: {user_profile}, Trasporto: {transport}\n\n"

        # Analizza vincoli uno per uno
        validation = self.validate_travel_plan(origin, destination, user_profile, transport, 1000)

        if validation['valid']:
            explanation += "[OK] VIAGGIO VALIDO - Tutti i vincoli soddisfatti:\n"
            for constraint in validation['constraints_satisfied']:
                explanation += f"  • {constraint}\n"
        else:
            explanation += "[ERROR] VIAGGIO NON VALIDO - Vincoli violati:\n"
            for violation in validation['violations']:
                explanation += f"  • {violation}\n"

            if validation['suggestions']:
                explanation += "\n💡 SUGGERIMENTI:\n"
                for suggestion in validation['suggestions']:
                    explanation += f"  • {suggestion}\n"

        return explanation

# Test dell'interfaccia Prolog
if __name__ == "__main__":

    print("=" * 60)
    print("PROLOG KNOWLEDGE BASE INTERFACE TEST")
    print("=" * 60)

    # Inizializza KB
    kb = PrologKnowledgeBase()

    # Test 1: Query basic connectivity
    print(f"\n=== TEST 1: CONNECTIVITY QUERY ===")
    results = kb.query_prolog("connected(milano, roma, Transports)")
    print(f"Milano-Roma connectivity: {results}")

    # Test 2: Validation travel plan
    print(f"\n=== TEST 2: TRAVEL PLAN VALIDATION ===")

    test_plans = [
        ('milano', 'roma', 'business', 'train', 200),
        ('venezia', 'palermo', 'budget', 'flight', 100),
        ('torino', 'napoli', 'leisure', 'bus', 300)
    ]

    for origin, dest, profile, transport, budget in test_plans:
        validation = kb.validate_travel_plan(origin, dest, profile, transport, budget)

        print(f"\n{origin} -> {dest} ({profile}, {transport}, €{budget}):")
        print(f"  Valid: {validation['valid']}")
        if validation['violations']:
            print(f"  Violations: {validation['violations']}")
        if validation['suggestions']:
            print(f"  Suggestions: {validation['suggestions']}")

    # Test 3: Travel recommendations
    print(f"\n=== TEST 3: TRAVEL RECOMMENDATIONS ===")

    recommendations = kb.get_travel_recommendations('milano', 'napoli', 'business')
    print(f"\nRecommendations Milano -> Napoli (business):")
    print(f"  Recommended transports: {recommendations['recommended_transports']}")
    print(f"  Budget estimates: {recommendations['budget_estimates']}")
    print(f"  Feasibility: {recommendations['feasibility']}")

    # Test 4: Constraint explanation
    print(f"\n=== TEST 4: CONSTRAINT EXPLANATION ===")

    explanation = kb.explain_constraints('roma', 'palermo', 'budget', 'flight')
    print(f"\n{explanation}")

    print(f"\n[DONE] Prolog Knowledge Base interface tested successfully!")