#!/usr/bin/env python3
"""
Sistema di pianificazione viaggi che integra quattro paradigmi dell'intelligenza artificiale:
- Search Algorithms (A*, Floyd-Warshall, Dijkstra)
- Machine Learning (Regression, Classification, Ensemble)
- Probabilistic Reasoning (Bayesian Networks)
- Logic Programming (Prolog Knowledge Base)
- Multi-paradigm Integration e Evaluation

Author: Antonio Colamartino
University: Università di Bari "Aldo Moro"
Course: Ingegneria della Conoscenza
"""

import argparse
import json
import sys
import os
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all system components
from data_collection.transport_data import CityGraph
from search_algorithms.pathfinder import AdvancedPathfinder, OptimizationObjective
from ml_models.ml_pathfinder_integration import MLEnhancedPathfinder
from bayesian_network.uncertainty_models import TravelUncertaintyNetwork
from prolog_kb.prolog_interface import PrologKnowledgeBase
from evaluation.comprehensive_evaluation import ComprehensiveEvaluationFramework

class IntelligentTravelPlanner:
    """
    Sistema di pianificazione viaggi intelligente - Orchestratore finale

    PARADIGMI INTEGRATI:
    1. Search Algorithms: A*, Floyd-Warshall, Dijkstra per pathfinding ottimale
    2. Machine Learning: Predizione prezzi, tempi, classificazione profili utente
    3. Probabilistic Reasoning: Bayesian Networks per gestione incertezza
    4. Logic Programming: Prolog KB per vincoli e regole di business
    5. Multi-paradigm Integration: Orchestrazione intelligente di tutti i componenti

    ARCHITETTURA SISTEMA:
    Input --> User Profiling (ML) --> Search (A*+ML) --> Validation (Prolog) -->
    Uncertainty (Bayes) --> Final Recommendation
    """

    def __init__(self,
                 training_mode: bool = True,
                 evaluation_mode: bool = False,
                 verbose: bool = True):

        self.verbose = verbose
        self.training_mode = training_mode
        self.evaluation_mode = evaluation_mode

        if self.verbose:
            print("=" * 80)
            print("INTELLIGENT TRAVEL PLANNER")
            print("=" * 80)
            print("Initializing all system components...")

        # Initialize core components
        self._initialize_components()

        # Train models if needed
        if training_mode:
            self._train_system()

        # Initialize evaluation framework if needed
        if evaluation_mode:
            self.evaluator = ComprehensiveEvaluationFramework()

        if self.verbose:
            print("[OK] System initialization completed successfully!")
            print("=" * 80)

    def _initialize_components(self):
        """
        Inizializza tutti i componenti del sistema
        """

        if self.verbose:
            print("\n[1/5] Initializing Graph and Search Algorithms...")

        # 1. Core graph and search algorithms
        self.city_graph = CityGraph()
        self.base_pathfinder = AdvancedPathfinder(self.city_graph)

        if self.verbose:
            print(f"   [OK] Graph loaded with {len(self.city_graph.cities)} cities")
            print(f"   [OK] {self.city_graph.graph.number_of_edges()} connections available")

        if self.verbose:
            print("\n[2/5] Initializing ML-Enhanced Pathfinder...")

        # 2. ML-enhanced pathfinding
        self.ml_pathfinder = MLEnhancedPathfinder(self.city_graph)

        if self.verbose:
            print("\n[3/5] Initializing Bayesian Network...")

        # 3. Bayesian network for uncertainty
        self.bayesian_net = TravelUncertaintyNetwork()

        if self.verbose:
            print(f"   [OK] Bayesian Network with {len(self.bayesian_net.nodes)} nodes")

        if self.verbose:
            print("\n[4/5] Initializing Prolog Knowledge Base...")

        # 4. Prolog knowledge base
        self.prolog_kb = PrologKnowledgeBase()

        if self.verbose:
            print("\n[5/5] Integration layer ready!")

        # 5. System state
        self.system_ready = False

    def _train_system(self):
        """
        Training di tutti i componenti ML del sistema
        """

        if self.verbose:
            print(f"\n{'='*20} SYSTEM TRAINING PHASE {'='*20}")

        # Train ML models
        training_results = self.ml_pathfinder.train_ml_models(n_scenarios=800)

        if self.verbose:
            print(f"\n[OK] ML Training completed:")
            price_r2 = training_results['price_results'][self.ml_pathfinder.price_predictor.best_model_name]['test_r2']
            user_acc = training_results['user_results'][self.ml_pathfinder.user_classifier.best_model_name]['test_accuracy']
            time_r2 = training_results['time_results']['cv_r2_mean']

            print(f"   • Price Predictor: R² = {price_r2:.3f}")
            print(f"   • User Classifier: Accuracy = {user_acc:.3f}")
            print(f"   • Time Estimator: R² = {time_r2:.3f} ± {training_results['time_results']['cv_r2_std']:.3f}")

        self.system_ready = True

    def plan_travel(self,
                   origin: str,
                   destination: str,
                   user_profile: Dict[str, Any] = None,
                   budget: float = 500.0,
                   season: str = 'summer',
                   weather_condition: str = 'Fair',
                   optimization_objectives: List[str] = None) -> Dict[str, Any]:
        """
        Pianificazione viaggio intelligente con integrazione multi-paradigma

        PIPELINE COMPLETA:
        1. User Profiling (ML Classification)
        2. Route Finding (ML-Enhanced A*)
        3. Constraint Validation (Prolog KB)
        4. Uncertainty Analysis (Bayesian Network)
        5. Final Recommendation (Integration)

        Args:
            origin: Città di partenza
            destination: Città di destinazione
            user_profile: Dati utente per personalizzazione
            budget: Budget massimo viaggio
            season: Stagione di viaggio
            weather_condition: Condizioni meteo previste
            optimization_objectives: Obiettivi ottimizzazione

        Returns:
            Dict completo con raccomandazione finale, analisi, spiegazioni
        """

        if not self.system_ready:
            raise RuntimeError("System not trained. Initialize with training_mode=True")

        start_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"INTELLIGENT TRAVEL PLANNING: {origin} --> {destination}")
            print(f"{'='*60}")

        travel_plan = {
            'request': {
                'origin': origin,
                'destination': destination,
                'user_profile': user_profile,
                'budget': budget,
                'season': season,
                'weather_condition': weather_condition
            },
            'analysis': {},
            'recommendations': {},
            'system_performance': {},
            'explanations': {}
        }

        # STEP 1: User Profiling & Personalization
        if self.verbose:
            print(f"\n[STEP 1] User Profiling & Personalization...")

        if user_profile:
            predicted_profile, profile_probs = self.ml_pathfinder.user_classifier.predict_user_profile(**user_profile)
            personalized_weights = self.ml_pathfinder.user_classifier.get_personalized_weights(predicted_profile, profile_probs)

            travel_plan['analysis']['user_profiling'] = {
                'predicted_profile': predicted_profile,
                'confidence': profile_probs[predicted_profile],
                'profile_probabilities': profile_probs,
                'personalized_weights': personalized_weights
            }

            if self.verbose:
                print(f"   Profile: {predicted_profile} (confidence: {profile_probs[predicted_profile]:.2f})")
                print(f"   Weights: Cost={personalized_weights['cost']:.2f}, Time={personalized_weights['time']:.2f}")
        else:
            predicted_profile = 'leisure'
            personalized_weights = {'cost': 0.35, 'time': 0.25, 'distance': 0.25, 'comfort': 0.15}

            travel_plan['analysis']['user_profiling'] = {
                'predicted_profile': predicted_profile,
                'confidence': 0.5,
                'note': 'Default profile used'
            }

        # STEP 2: Multi-Algorithm Route Finding
        if self.verbose:
            print(f"\n[STEP 2] Multi-Algorithm Route Finding...")

        route_options = {}

        # 2a. Base A* (Baseline)
        try:
            base_route = self.base_pathfinder.multi_objective_astar(
                origin, destination,
                objectives=[OptimizationObjective.DISTANCE, OptimizationObjective.TIME, OptimizationObjective.COST],
                weights=[0.33, 0.33, 0.34]
            )

            if base_route:
                route_options['base_astar'] = {
                    'path': base_route.path,
                    'total_cost': base_route.total_cost,
                    'total_time': base_route.total_time,
                    'total_distance': base_route.total_distance,
                    'score': base_route.normalized_score
                }

                if self.verbose:
                    print(f"   Base A*: {' --> '.join(base_route.path[:3])}{'...' if len(base_route.path) > 3 else ''}")
                    print(f"            Cost: €{base_route.total_cost:.2f}, Time: {base_route.total_time:.1f}h")
        except Exception as e:
            if self.verbose:
                print(f"   Base A* failed: {e}")

        # 2b. ML-Enhanced A* (Main recommendation)
        try:
            ml_route, ml_metadata = self.ml_pathfinder.find_ml_enhanced_route(
                origin, destination,
                user_profile=user_profile,
                season=season,
                travel_context={'weather_condition': weather_condition, 'budget': budget}
            )

            if ml_route:
                route_options['ml_enhanced'] = {
                    'path': ml_route.path,
                    'total_cost': ml_route.total_cost,
                    'total_time': ml_route.total_time,
                    'total_distance': ml_route.total_distance,
                    'score': ml_route.normalized_score,
                    'metadata': ml_metadata
                }

                if self.verbose:
                    print(f"   ML-Enhanced: {' --> '.join(ml_route.path[:3])}{'...' if len(ml_route.path) > 3 else ''}")
                    print(f"                Cost: €{ml_route.total_cost:.2f}, Time: {ml_route.total_time:.1f}h")
        except Exception as e:
            if self.verbose:
                print(f"   ML-Enhanced failed: {e}")

        # 2c. Floyd-Warshall (Optimal distance)
        try:
            fw_distance = self.base_pathfinder.all_distances.get(origin, {}).get(destination, float('inf'))
            fw_path = self.base_pathfinder.get_floyd_warshall_path(origin, destination)

            if fw_path and fw_distance != float('inf'):
                route_options['floyd_warshall'] = {
                    'path': fw_path,
                    'total_distance': fw_distance,
                    'note': 'Optimal distance path'
                }

                if self.verbose:
                    print(f"   Floyd-Warshall: {fw_distance:.0f}km optimal distance")
        except Exception as e:
            if self.verbose:
                print(f"   Floyd-Warshall failed: {e}")

        travel_plan['analysis']['route_finding'] = route_options

        # STEP 3: Constraint Validation (Prolog KB)
        if self.verbose:
            print(f"\n[STEP 3] Constraint Validation (Prolog KB)...")

        # Use best route for validation (ML-enhanced if available, else base)
        best_route_key = 'ml_enhanced' if 'ml_enhanced' in route_options else 'base_astar'
        best_route = route_options.get(best_route_key)

        if best_route:
            # Determine primary transport (simplified)
            primary_transport = 'train'  # Default assumption

            validation = self.prolog_kb.validate_travel_plan(
                origin, destination, predicted_profile, primary_transport, budget, season
            )

            travel_plan['analysis']['constraint_validation'] = validation

            if self.verbose:
                print(f"   Validation: {'[OK] Valid' if validation['valid'] else '[ERROR] Issues found'}")
                if validation['violations']:
                    print(f"   Violations: {', '.join(validation['violations'])}")
        else:
            travel_plan['analysis']['constraint_validation'] = {'error': 'No route to validate'}

        # STEP 4: Uncertainty Analysis (Bayesian Network)
        if self.verbose:
            print(f"\n[STEP 4] Uncertainty Analysis (Bayesian Network)...")

        try:
            # Map weather condition to Bayesian network format
            weather_map = {'Good': 'Good', 'Fair': 'Fair', 'Bad': 'Bad'}
            weather_bn = weather_map.get(weather_condition, 'Fair')

            uncertainty_analysis = {}

            for transport in ['Train', 'Bus', 'Flight']:
                prediction = self.bayesian_net.predict_trip_outcome(transport, weather_bn, season.title())
                uncertainty_analysis[transport.lower()] = {
                    'success_probability': prediction['trip_success']['Success'],
                    'delay_probability': prediction['delays']['Major'] + prediction['delays']['Minor'],
                    'satisfaction_probability': prediction['user_satisfaction']['High']
                }

            travel_plan['analysis']['uncertainty_analysis'] = uncertainty_analysis

            if self.verbose:
                for transport, probs in uncertainty_analysis.items():
                    print(f"   {transport.title()}: Success={probs['success_probability']:.2f}, "
                          f"Delays={probs['delay_probability']:.2f}")

        except Exception as e:
            travel_plan['analysis']['uncertainty_analysis'] = {'error': str(e)}
            if self.verbose:
                print(f"   Bayesian analysis failed: {e}")

        # STEP 5: Final Recommendation & Integration
        if self.verbose:
            print(f"\n[STEP 5] Final Recommendation & Integration...")

        # Select best recommendation
        if 'ml_enhanced' in route_options:
            recommended_route = route_options['ml_enhanced']
            recommendation_source = 'ML-Enhanced A* with personalization'
        elif 'base_astar' in route_options:
            recommended_route = route_options['base_astar']
            recommendation_source = 'Base A* algorithm'
        else:
            recommended_route = None
            recommendation_source = 'No valid route found'

        travel_plan['recommendations']['primary'] = {
            'route': recommended_route,
            'source': recommendation_source,
            'confidence': 0.85 if 'ml_enhanced' in route_options else 0.65
        }

        # Alternative recommendations
        alternatives = []
        for route_name, route_data in route_options.items():
            if route_name != best_route_key and 'path' in route_data:
                alternatives.append({
                    'method': route_name,
                    'route': route_data,
                    'note': f"Alternative using {route_name}"
                })

        travel_plan['recommendations']['alternatives'] = alternatives

        # STEP 6: Performance metrics & Explanations
        computation_time = time.time() - start_time

        travel_plan['system_performance'] = {
            'total_computation_time': computation_time,
            'components_used': ['ML', 'Search', 'Prolog', 'Bayesian'],
            'success': recommended_route is not None
        }

        # Generate explanations
        travel_plan['explanations'] = self._generate_explanations(travel_plan)

        if self.verbose:
            print(f"\n[OK] Travel planning completed in {computation_time:.2f}s")
            if recommended_route:
                print(f"   Recommended: {' --> '.join(recommended_route['path'])}")
                print(f"   Cost: €{recommended_route.get('total_cost', 0):.2f}, "
                      f"Time: {recommended_route.get('total_time', 0):.1f}h")

        return travel_plan

    def _generate_explanations(self, travel_plan: Dict[str, Any]) -> Dict[str, str]:
        """
        Genera spiegazioni human-readable del processo decisionale

         AI spiegabile
        """

        explanations = {}

        # User profiling explanation
        if 'user_profiling' in travel_plan['analysis']:
            profiling = travel_plan['analysis']['user_profiling']
            profile = profiling.get('predicted_profile', 'unknown')
            confidence = profiling.get('confidence', 0)

            explanations['user_profiling'] = (
                f"Based on your preferences, you've been classified as a '{profile}' traveler "
                f"with {confidence:.0%} confidence. This means the system will prioritize "
                f"routes that match typical {profile} preferences."
            )

        # Route selection explanation
        if 'route_finding' in travel_plan['analysis']:
            route_count = len(travel_plan['analysis']['route_finding'])
            explanations['route_selection'] = (
                f"The system evaluated {route_count} different routing algorithms to find "
                f"the best path. The ML-enhanced algorithm was preferred because it adapts "
                f"to your specific profile and provides more accurate cost/time predictions."
            )

        # Constraint validation explanation
        if 'constraint_validation' in travel_plan['analysis']:
            validation = travel_plan['analysis']['constraint_validation']
            if validation.get('valid', False):
                explanations['constraints'] = (
                    "All logical constraints were satisfied: the route is feasible within "
                    "your budget, suitable for your profile, and has no seasonal restrictions."
                )
            else:
                violations = validation.get('violations', [])
                explanations['constraints'] = (
                    f"Some constraints were not satisfied: {', '.join(violations)}. "
                    f"Consider adjusting your budget or travel dates."
                )

        # Uncertainty analysis explanation
        if 'uncertainty_analysis' in travel_plan['analysis']:
            explanations['uncertainty'] = (
                "The system analyzed weather impact and transport reliability using "
                "probabilistic reasoning. Flight delays are more likely in bad weather, "
                "while trains and buses are generally more reliable."
            )

        return explanations

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """
        Esegue valutazione completa del sistema

        Returns:
            Risultati completi con metriche compliant
        """

        if not hasattr(self, 'evaluator'):
            self.evaluator = ComprehensiveEvaluationFramework()

        print(f"\n{'='*20} COMPREHENSIVE EVALUATION {'='*20}")

        # Generate evaluation dataset
        self.evaluator.generate_evaluation_dataset(n_scenarios=800)

        # Run evaluation
        results = self.evaluator.evaluate_all_systems()

        # Generate tables
        tables = self.evaluator.generate_results_tables()

        # Save results
        self.evaluator.save_evaluation_results()

        # Print summary
        self.evaluator.print_summary_report()

        return {
            'raw_results': results,
            'formatted_tables': tables,
            'evaluation_completed': True
        }

    def interactive_demo(self):
        """
        Demo interattivo del sistema
        """

        print(f"\n{'='*60}")
        print("INTERACTIVE TRAVEL PLANNING DEMO")
        print(f"{'='*60}")

        # Predefined demo scenarios
        demo_scenarios = [
            {
                'name': 'Business Trip',
                'origin': 'milano',
                'destination': 'roma',
                'user_profile': {
                    'user_age': 35,
                    'user_income': 70000,
                    'price_sensitivity': 0.2,
                    'time_priority': 0.9,
                    'comfort_priority': 0.8
                },
                'budget': 300,
                'season': 'summer'
            },
            {
                'name': 'Budget Travel',
                'origin': 'venezia',
                'destination': 'napoli',
                'user_profile': {
                    'user_age': 22,
                    'user_income': 25000,
                    'price_sensitivity': 0.95,
                    'time_priority': 0.2,
                    'comfort_priority': 0.3
                },
                'budget': 80,
                'season': 'spring'
            },
            {
                'name': 'Leisure Travel',
                'origin': 'torino',
                'destination': 'palermo',
                'user_profile': {
                    'user_age': 28,
                    'user_income': 45000,
                    'price_sensitivity': 0.6,
                    'time_priority': 0.4,
                    'comfort_priority': 0.6
                },
                'budget': 400,
                'season': 'autumn'
            }
        ]

        for i, scenario in enumerate(demo_scenarios, 1):
            print(f"\n[DEMO {i}/3] {scenario['name']}")
            print("-" * 40)

            try:
                result = self.plan_travel(**scenario)

                # Print summary
                if result['recommendations']['primary']['route']:
                    route = result['recommendations']['primary']['route']
                    print(f"[OK] Route found: {' --> '.join(route['path'])}")
                    print(f"   Cost: €{route.get('total_cost', 0):.2f}")
                    print(f"   Time: {route.get('total_time', 0):.1f} hours")
                    print(f"   Confidence: {result['recommendations']['primary']['confidence']:.0%}")

                    # Print explanation
                    if 'user_profiling' in result['explanations']:
                        print(f"   {result['explanations']['user_profiling']}")
                else:
                    print("[ERROR] No suitable route found")

            except Exception as e:
                print(f"[ERROR] Demo failed: {e}")

        print(f"\n{'='*60}")
        print("DEMO COMPLETED")
        print(f"{'='*60}")

def main():
    """
    Main entry point con CLI interface
    """

    parser = argparse.ArgumentParser(
        description="Intelligent Travel Planner - ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python intelligent_travel_planner.py --demo
  python intelligent_travel_planner.py --evaluate
  python intelligent_travel_planner.py --plan milano roma --budget 200
  python intelligent_travel_planner.py --plan venezia napoli --profile business --budget 300
        """
    )

    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    parser.add_argument('--evaluate', action='store_true', help='Run comprehensive evaluation')
    parser.add_argument('--no-training', action='store_true', help='Skip ML model training')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')

    # Travel planning arguments
    parser.add_argument('--plan', nargs=2, metavar=('ORIGIN', 'DESTINATION'), help='Plan travel route')
    parser.add_argument('--budget', type=float, default=500.0, help='Travel budget (default: 500)')
    parser.add_argument('--season', choices=['summer', 'winter', 'spring', 'autumn'],
                       default='summer', help='Travel season (default: summer)')
    parser.add_argument('--profile', choices=['business', 'leisure', 'budget'],
                       help='User profile for personalization')
    parser.add_argument('--weather', choices=['Good', 'Fair', 'Bad'],
                       default='Fair', help='Weather condition (default: Fair)')

    args = parser.parse_args()

    # Initialize system
    try:
        planner = IntelligentTravelPlanner(
            training_mode=not args.no_training,
            evaluation_mode=args.evaluate,
            verbose=not args.quiet
        )

        # Run requested operation
        if args.demo:
            planner.interactive_demo()

        elif args.evaluate:
            planner.run_comprehensive_evaluation()

        elif args.plan:
            origin, destination = args.plan

            # Build user profile if specified
            user_profile = None
            if args.profile:
                user_profile = {
                    'user_age': 30,
                    'user_income': 50000 if args.profile == 'business' else 30000 if args.profile == 'leisure' else 20000,
                    'price_sensitivity': 0.2 if args.profile == 'business' else 0.6 if args.profile == 'leisure' else 0.9,
                    'time_priority': 0.9 if args.profile == 'business' else 0.5 if args.profile == 'leisure' else 0.2,
                    'comfort_priority': 0.8 if args.profile == 'business' else 0.6 if args.profile == 'leisure' else 0.3
                }

            # Plan travel
            result = planner.plan_travel(
                origin=origin,
                destination=destination,
                user_profile=user_profile,
                budget=args.budget,
                season=args.season,
                weather_condition=args.weather
            )

            # Print results
            print(f"\n{'='*60}")
            print("TRAVEL PLANNING RESULT")
            print(f"{'='*60}")

            if result['recommendations']['primary']['route']:
                route = result['recommendations']['primary']['route']
                print(f"[OK] Recommended Route: {' --> '.join(route['path'])}")
                print(f"   Total Cost: €{route.get('total_cost', 0):.2f}")
                print(f"   Total Time: {route.get('total_time', 0):.1f} hours")
                print(f"   Distance: {route.get('total_distance', 0):.0f} km")
                print(f"   Confidence: {result['recommendations']['primary']['confidence']:.0%}")
                print(f"   Method: {result['recommendations']['primary']['source']}")

                # Print explanations
                print(f"\nEXPLANATIONS:")
                for key, explanation in result['explanations'].items():
                    print(f"   • {explanation}")

            else:
                print("[ERROR] No suitable route found for your criteria")

                # Print reasons
                if 'constraint_validation' in result['analysis']:
                    validation = result['analysis']['constraint_validation']
                    if validation.get('violations'):
                        print(f"   Issues: {', '.join(validation['violations'])}")

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()