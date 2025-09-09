import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.transport_data import CityGraph
from search_algorithms.pathfinder import AdvancedPathfinder, OptimizationObjective, TravelRoute, RouteSegment
from ml_models.dataset_generator import TravelDatasetGenerator
from ml_models.predictor_models import TravelPricePredictor, TravelTimeEstimator
from ml_models.preference_classifier import UserPreferenceClassifier

class MLEnhancedPathfinder:
    """
    Sistema di ricerca IBRIDO: Algoritmi + Machine Learning

    INTEGRAZIONE PARADIGMI ICon:
    1. Search Algorithms (A*, Floyd-Warshall) + ML Predictions
    2. User Profiling --> Personalized Heuristics
    3. Dynamic Pricing --> Real-time Cost Functions
    4. Uncertainty Handling --> Probabilistic Estimates

    CONTRIBUTI ORIGINALI:
    - Euristiche A* guidate da ML predictions
    - Pesi multi-obiettivo adattivi basati su profilo utente
    - Cost functions dinamiche con pricing model
    - Valutazione comparativa Algoritmi vs ML+Algoritmi
    """

    def __init__(self, city_graph: CityGraph):
        self.city_graph = city_graph
        self.base_pathfinder = AdvancedPathfinder(city_graph)

        # Componenti ML
        self.price_predictor = TravelPricePredictor()
        self.time_estimator = TravelTimeEstimator()
        self.user_classifier = UserPreferenceClassifier()

        # Status training
        self.models_trained = False

    def train_ml_models(self, dataset_path: str = None, n_scenarios: int = 1000):
        """
        Training completo di tutti i modelli ML

        SETUP VALUTAZIONE ICon:
        - Dataset sintetico realistico
        - Cross-validation K-fold
        - Multiple runs con statistiche
        """

        print("=" * 60)
        print("ML-ENHANCED PATHFINDER: TRAINING PHASE")
        print("=" * 60)

        # Genera o carica dataset
        if dataset_path and os.path.exists(dataset_path):
            print(f"[DATA] Caricando dataset da: {dataset_path}")
            df = pd.read_csv(dataset_path)
        else:
            print(f"[DATA] Generando nuovo dataset ({n_scenarios} scenarios)...")
            generator = TravelDatasetGenerator(self.city_graph)
            df = generator.generate_travel_scenarios(n_scenarios)

            # Salva per riuso
            output_path = "data/ml_training_dataset.csv"
            os.makedirs("data", exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"[SAVED] Dataset salvato in: {output_path}")

        print(f"[DATA] Dataset shape: {df.shape}")
        print(f"[DATA] Target distributions:")
        print(f"   User profiles: {df['user_profile'].value_counts().to_dict()}")
        print(f"   Transport types: {df['transport_type'].value_counts().to_dict()}")

        # 1. TRAIN PRICE PREDICTOR
        print(f"\n{'='*20} PRICE PREDICTOR TRAINING {'='*20}")
        X_price, y_price = self.price_predictor.prepare_features(df)
        price_results = self.price_predictor.train_models(X_price, y_price)

        # 2. TRAIN TIME ESTIMATOR
        print(f"\n{'='*20} TIME ESTIMATOR TRAINING {'='*20}")
        X_time, y_time = self.time_estimator.prepare_features(df)
        time_results = self.time_estimator.train(X_time, y_time)

        # 3. TRAIN USER CLASSIFIER
        print(f"\n{'='*20} USER CLASSIFIER TRAINING {'='*20}")
        X_user, y_user = self.user_classifier.prepare_features(df)
        user_results = self.user_classifier.train_models(X_user, y_user)

        self.models_trained = True

        # Risultati complessivi
        print(f"\n{'='*20} TRAINING SUMMARY {'='*20}")
        print(f"Price Predictor - Best: {self.price_predictor.best_model_name} (R² = {price_results[self.price_predictor.best_model_name]['test_r2']:.3f})")
        print(f"Time Estimator  - R²: {time_results['cv_r2_mean']:.3f} ± {time_results['cv_r2_std']:.3f}")
        print(f"User Classifier - Best: {self.user_classifier.best_model_name} (Acc = {user_results[self.user_classifier.best_model_name]['test_accuracy']:.3f})")

        return {
            'price_results': price_results,
            'time_results': time_results,
            'user_results': user_results
        }

    def find_ml_enhanced_route(self,
                              origin: str,
                              destination: str,
                              user_profile: Dict = None,
                              season: str = 'summer',
                              travel_context: Dict = None) -> Tuple[TravelRoute, Dict]:
        """
        Ricerca percorso con ML-Enhanced Heuristics

        INNOVAZIONI RISPETTO A* BASE:
        1. Cost Functions dinamiche (ML price prediction)
        2. Heuristics personalizzate (user profiling)
        3. Multi-objective weights adattivi
        4. Uncertainty integration (prediction confidence)
        """

        if not self.models_trained:
            print("[WARNING] Modelli non trainati! Usando fallback algorithms...")
            return self.base_pathfinder.multi_objective_astar(origin, destination), {}

        print(f"\n[ML-A*] Enhanced pathfinding: {origin} -> {destination}")

        # 1. USER PROFILING & PERSONALIZATION
        if user_profile:
            predicted_profile, profile_probs = self.user_classifier.predict_user_profile(**user_profile)
            personalized_weights = self.user_classifier.get_personalized_weights(predicted_profile, profile_probs)
            print(f"   User Profile: {predicted_profile} (confidence: {profile_probs[predicted_profile]:.2f})")
        else:
            predicted_profile = 'leisure'
            personalized_weights = {'distance': 0.25, 'time': 0.25, 'cost': 0.35, 'comfort': 0.15}
            print(f"   User Profile: default leisure")

        # 2. CONTEXT PREPARATION
        context = travel_context or {}
        context.update({
            'season': season,
            'user_profile': predicted_profile,
            'is_weekend': context.get('is_weekend', False)
        })

        # 3. ML-ENHANCED A* SEARCH
        route = self._ml_astar_search(origin, destination, personalized_weights, context)

        # 4. PERFORMANCE METADATA
        metadata = {
            'user_profile': predicted_profile,
            'personalized_weights': personalized_weights,
            'ml_predictions_used': True,
            'search_algorithm': 'ML-Enhanced A*'
        }

        return route, metadata

    def _ml_astar_search(self,
                        origin: str,
                        destination: str,
                        weights: Dict[str, float],
                        context: Dict) -> Optional[TravelRoute]:
        """
        A* con euristiche ML-enhanced

        DIFFERENZE DA A* BASE:
        - g_score usa ML price/time prediction
        - h_score personalizzato per profilo utente
        - Pruning basato su confidence ML
        """

        from heapq import heappush, heappop

        # Priority queue: (f_score, g_score, current_city, path, route_segments)
        open_set = [(0, 0, origin, [origin], [])]
        closed_set = set()

        while open_set:
            f_score, g_score, current, path, route_segments = heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)

            # Goal raggiunto?
            if current == destination:
                return self.base_pathfinder._build_travel_route(route_segments, path)

            # Esplora vicini con ML predictions
            for neighbor in self.city_graph.graph.neighbors(current):
                if neighbor in closed_set:
                    continue

                # ML-Enhanced segment evaluation
                segment = self._evaluate_segment_with_ml(current, neighbor, context)
                if segment is None:
                    continue

                # Nuovo g_score con ML predictions
                new_route_segments = route_segments + [segment]
                new_g_score = self._calculate_ml_cost(new_route_segments, weights)

                # ML-Enhanced heuristic
                h_score = self._ml_heuristic(neighbor, destination, weights, context)

                new_f_score = new_g_score + h_score
                new_path = path + [neighbor]

                heappush(open_set, (new_f_score, new_g_score, neighbor, new_path, new_route_segments))

        return None

    def _evaluate_segment_with_ml(self, city1: str, city2: str, context: Dict) -> Optional[RouteSegment]:
        """
        Valutazione segmento con ML predictions invece di formule statiche
        """

        # Distanza base dal grafo
        try:
            edge_data = self.city_graph.graph[city1][city2]
            distance = edge_data['distance']
        except:
            return None

        # Trova miglior trasporto con ML predictions
        available_transports = []

        for transport in ['train', 'bus', 'flight']:
            if edge_data.get(transport, False):

                # ML PREDICTIONS invece di calcoli statici
                predicted_price = self.price_predictor.predict_price(
                    distance=distance,
                    origin=city1,
                    destination=city2,
                    transport_type=transport,
                    season=context.get('season', 'summer'),
                    user_profile=context.get('user_profile', 'leisure'),
                    is_weekend=context.get('is_weekend', False)
                )

                predicted_time = self.time_estimator.predict_time(
                    distance=distance,
                    transport_type=transport,
                    weather_factor=context.get('weather_factor', 1.0),
                    is_peak_hour=context.get('is_peak_hour', False)
                )

                # Comfort statico (potrebbe essere ML in futuro)
                comfort = {'train': 0.8, 'bus': 0.4, 'flight': 0.6}[transport]

                available_transports.append(RouteSegment(
                    origin=city1,
                    destination=city2,
                    transport_type=transport,
                    distance=distance,
                    time=predicted_time,
                    cost=predicted_price,
                    comfort=comfort,
                    departure_time=9  # Simplified
                ))

        # Restituisci miglior trasporto
        if available_transports:
            # Selezione basata su profilo utente (weighted score)
            weights = context.get('weights', {'cost': 0.4, 'time': 0.4, 'comfort': 0.2})

            def transport_score(t):
                return (weights.get('cost', 0.4) * (1.0 / (1.0 + t.cost / 100)) +
                       weights.get('time', 0.4) * (1.0 / (1.0 + t.time)) +
                       weights.get('comfort', 0.2) * t.comfort)

            return max(available_transports, key=transport_score)

        return None

    def _calculate_ml_cost(self, segments: List[RouteSegment], weights: Dict[str, float]) -> float:
        """
        Cost function multi-obiettivo con pesi personalizzati
        """
        if not segments:
            return 0

        total_distance = sum(s.distance for s in segments)
        total_time = sum(s.time for s in segments)
        total_cost = sum(s.cost for s in segments)
        avg_comfort = sum(s.comfort for s in segments) / len(segments)

        # Weighted sum con pesi personalizzati
        weighted_cost = (
            weights.get('distance', 0.25) * total_distance / 1000 +
            weights.get('time', 0.25) * total_time +
            weights.get('cost', 0.35) * total_cost / 100 +
            weights.get('comfort', 0.15) * (1.0 / avg_comfort)
        )

        return weighted_cost

    def _ml_heuristic(self, current: str, destination: str,
                     weights: Dict[str, float], context: Dict) -> float:
        """
        Euristica A* personalizzata con ML insight
        """

        # Base: distanza euclidea
        euclidean_dist = self.city_graph._calculate_real_distance(current, destination)

        # ML-informed estimates per lower bounds
        estimated_min_cost = euclidean_dist * 0.08  # Bus price minimum
        estimated_min_time = euclidean_dist / 500   # Flight speed maximum

        # Weighted heuristic basato su profilo utente
        weighted_heuristic = (
            weights.get('distance', 0.25) * euclidean_dist / 1000 +
            weights.get('time', 0.25) * estimated_min_time +
            weights.get('cost', 0.35) * estimated_min_cost / 100 +
            weights.get('comfort', 0.15) * 0.1  # Comfort penalty placeholder
        )

        return weighted_heuristic

    def comparative_evaluation(self, test_routes: List[Tuple[str, str]],
                             user_profiles: List[Dict] = None) -> pd.DataFrame:
        """
        Valutazione comparativa: Base Algorithms vs ML-Enhanced

        REQUIREMENT: Confronto quantitativo con baseline
        """

        print(f"\n{'='*20} COMPARATIVE EVALUATION {'='*20}")
        print(f"Testing {len(test_routes)} routes with multiple algorithms...")

        results = []

        for i, (origin, dest) in enumerate(test_routes):
            print(f"\n[{i+1}/{len(test_routes)}] Testing route: {origin} -> {dest}")

            # Test con diversi profili utente
            test_profiles = user_profiles or [
                {'user_age': 35, 'user_income': 60000, 'price_sensitivity': 0.2, 'time_priority': 0.9, 'comfort_priority': 0.8},  # business
                {'user_age': 28, 'user_income': 32000, 'price_sensitivity': 0.75, 'time_priority': 0.4, 'comfort_priority': 0.6}, # leisure
                {'user_age': 22, 'user_income': 20000, 'price_sensitivity': 0.95, 'time_priority': 0.2, 'comfort_priority': 0.3}  # budget
            ]

            for profile in test_profiles:

                # 1. BASELINE: Standard multi-objective A*
                try:
                    base_route = self.base_pathfinder.multi_objective_astar(
                        origin, dest,
                        objectives=[OptimizationObjective.DISTANCE, OptimizationObjective.TIME, OptimizationObjective.COST],
                        weights=[0.33, 0.33, 0.34]
                    )
                except:
                    base_route = None

                # 2. ML-ENHANCED
                try:
                    ml_route, ml_metadata = self.find_ml_enhanced_route(
                        origin, dest, user_profile=profile
                    )
                except:
                    ml_route, ml_metadata = None, {}

                # 3. FLOYD-WARSHALL (distance-only baseline)
                fw_distance = self.base_pathfinder.all_distances.get(origin, {}).get(dest, float('inf'))
                fw_path = self.base_pathfinder.get_floyd_warshall_path(origin, dest)

                # Raccolta risultati
                result = {
                    'origin': origin,
                    'destination': dest,
                    'user_profile': ml_metadata.get('user_profile', 'unknown'),
                    'user_income': profile.get('user_income', 0),

                    # Base A* results
                    'base_total_cost': base_route.total_cost if base_route else None,
                    'base_total_time': base_route.total_time if base_route else None,
                    'base_total_distance': base_route.total_distance if base_route else None,
                    'base_path_length': len(base_route.path) if base_route else None,
                    'base_score': base_route.normalized_score if base_route else None,

                    # ML-Enhanced results
                    'ml_total_cost': ml_route.total_cost if ml_route else None,
                    'ml_total_time': ml_route.total_time if ml_route else None,
                    'ml_total_distance': ml_route.total_distance if ml_route else None,
                    'ml_path_length': len(ml_route.path) if ml_route else None,
                    'ml_score': ml_route.normalized_score if ml_route else None,

                    # Floyd-Warshall baseline
                    'fw_distance': fw_distance if fw_distance != float('inf') else None,
                    'fw_path_length': len(fw_path) if fw_path else None,

                    # Improvements
                    'cost_improvement_pct': None,
                    'time_improvement_pct': None,
                    'score_improvement_pct': None
                }

                # Calcola miglioramenti percentuali
                if base_route and ml_route:
                    if base_route.total_cost > 0:
                        result['cost_improvement_pct'] = ((base_route.total_cost - ml_route.total_cost) / base_route.total_cost) * 100
                    if base_route.total_time > 0:
                        result['time_improvement_pct'] = ((base_route.total_time - ml_route.total_time) / base_route.total_time) * 100
                    if base_route.normalized_score > 0:
                        result['score_improvement_pct'] = ((ml_route.normalized_score - base_route.normalized_score) / base_route.normalized_score) * 100

                results.append(result)

        # DataFrame risultati
        df_results = pd.DataFrame(results)

        # Summary statistics (ICon requirement)
        print(f"\n{'='*20} EVALUATION RESULTS {'='*20}")

        if len(df_results) > 0:
            print(f"Total test cases: {len(df_results)}")

            # Metriche comparative
            valid_cases = df_results.dropna(subset=['cost_improvement_pct', 'time_improvement_pct', 'score_improvement_pct'])

            if len(valid_cases) > 0:
                print(f"\nML vs Base A* Performance:")
                print(f"   Cost Improvement: {valid_cases['cost_improvement_pct'].mean():.2f}% ± {valid_cases['cost_improvement_pct'].std():.2f}%")
                print(f"   Time Improvement: {valid_cases['time_improvement_pct'].mean():.2f}% ± {valid_cases['time_improvement_pct'].std():.2f}%")
                print(f"   Score Improvement: {valid_cases['score_improvement_pct'].mean():.2f}% ± {valid_cases['score_improvement_pct'].std():.2f}%")

                # Success rate
                cost_wins = (valid_cases['cost_improvement_pct'] > 0).sum()
                time_wins = (valid_cases['time_improvement_pct'] > 0).sum()
                score_wins = (valid_cases['score_improvement_pct'] > 0).sum()

                print(f"\nWin Rates (ML better than Base):")
                print(f"   Cost: {cost_wins}/{len(valid_cases)} ({cost_wins/len(valid_cases)*100:.1f}%)")
                print(f"   Time: {time_wins}/{len(valid_cases)} ({time_wins/len(valid_cases)*100:.1f}%)")
                print(f"   Overall Score: {score_wins}/{len(valid_cases)} ({score_wins/len(valid_cases)*100:.1f}%)")

        return df_results

    def save_models(self, model_dir: str = "ml_models/trained/"):
        """Salva tutti i modelli trainati"""
        os.makedirs(model_dir, exist_ok=True)

        self.price_predictor.save_model(f"{model_dir}/price_predictor.pkl")
        self.user_classifier.save_model(f"{model_dir}/user_classifier.pkl")

        print(f"[SAVED] Tutti i modelli salvati in: {model_dir}")

    def load_models(self, model_dir: str = "ml_models/trained/"):
        """Carica modelli pre-trainati"""

        self.price_predictor.load_model(f"{model_dir}/price_predictor.pkl")
        self.user_classifier.load_model(f"{model_dir}/user_classifier.pkl")

        self.models_trained = True
        print(f"[LOADED] Tutti i modelli caricati da: {model_dir}")

# Test completo del sistema integrato
if __name__ == "__main__":

    print("=" * 80)
    print("ML-ENHANCED PATHFINDER: COMPLETE INTEGRATION TEST")
    print("=" * 80)

    # Setup
    city_graph = CityGraph()
    ml_pathfinder = MLEnhancedPathfinder(city_graph)

    # Training
    training_results = ml_pathfinder.train_ml_models(n_scenarios=800)

    # Test routes per evaluation
    test_routes = [
        ('Milano', 'Roma'),
        ('Venezia', 'Napoli'),
        ('Torino', 'Bari'),
        ('Bologna', 'Palermo'),
        ('Firenze', 'Cagliari')
    ]

    # Comparative evaluation
    evaluation_df = ml_pathfinder.comparative_evaluation(test_routes)

    # Salva risultati
    os.makedirs('data', exist_ok=True)
    evaluation_df.to_csv('data/ml_pathfinder_evaluation.csv', index=False)
    print(f"\n[SAVED] Evaluation results saved to: data/ml_pathfinder_evaluation.csv")

    # Test singolo dettagliato
    print(f"\n{'='*20} DETAILED SINGLE TEST {'='*20}")

    origin, destination = 'Milano', 'Roma'
    user_profile = {
        'user_age': 32,
        'user_income': 45000,
        'price_sensitivity': 0.6,
        'time_priority': 0.7,
        'comfort_priority': 0.5
    }

    ml_route, metadata = ml_pathfinder.find_ml_enhanced_route(
        origin, destination,
        user_profile=user_profile,
        season='summer'
    )

    if ml_route:
        print(f"\nML-Enhanced Route: {origin} -> {destination}")
        print(f"   Path: {' -> '.join(ml_route.path)}")
        print(f"   Total Cost: €{ml_route.total_cost:.2f}")
        print(f"   Total Time: {ml_route.total_time:.1f} hours")
        print(f"   Distance: {ml_route.total_distance:.0f} km")
        print(f"   Comfort Score: {ml_route.avg_comfort:.2f}")
        print(f"   Overall Score: {ml_route.normalized_score:.3f}")
        print(f"   User Profile: {metadata['user_profile']}")
        print(f"   Personalized Weights: {metadata['personalized_weights']}")

    print(f"\n[DONE] ML-Enhanced Pathfinder integration complete!")