import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
import itertools
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class BayesianNode:
    """
    Nodo della Rete Bayesiana con tabella probabilità condizionale

    CONCETTI:
    - Conditional Independence: P(X|Y,Z) dove Y,Z sono genitori
    - Probability Tables: CPT per ogni combinazione genitori
    - Local Markov Property: nodo indipendente da non-discendenti dati genitori
    """

    name: str
    domain: List[str]
    parents: List[str] = None
    cpt: Dict = None  # Conditional Probability Table

    def __post_init__(self):
        if self.parents is None:
            self.parents = []
        if self.cpt is None:
            self.cpt = {}

class TravelUncertaintyNetwork:
    """
    Rete Bayesiana per modellazione incertezza in travel planning

    ARGOMENTI DEL PROGRAMMA IMPLEMENTATI:
    1. Probabilistic Reasoning: inferenza sotto incertezza
    2. Causal Models: relazioni causa-effetto tra variabili
    3. Bayesian Inference: calcolo probabilità posteriori
    4. Decision Theory: scelte ottimali sotto incertezza
    5. Graphical Models: rappresentazione compatta dipendenze

    STRUTTURA RETE:
    Weather --> Transport_Delays --> Trip_Success
    Season --> Weather
    Transport_Type --> Transport_Delays
    User_Budget --> User_Satisfaction
    Trip_Cost --> User_Satisfaction
    """

    def __init__(self):
        self.nodes = {}
        self._build_travel_network()

    def _build_travel_network(self):
        """
        Costruisce rete Bayesiana per incertezza viaggi

        KNOWLEDGE ENGINEERING:
        - Domain expertise --> structure network
        - Historical data --> probability parameters
        - Causal relationships --> directed edges
        """

        print("[BAYES] Costruendo rete Bayesiana per incertezza viaggi...")

        # 1. WEATHER NODE (root node)
        weather_node = BayesianNode(
            name="Weather",
            domain=["Good", "Fair", "Bad"],
            parents=[],
            cpt={
                (): {"Good": 0.6, "Fair": 0.25, "Bad": 0.15}
            }
        )

        # 2. SEASON NODE (root node)
        season_node = BayesianNode(
            name="Season",
            domain=["Summer", "Winter", "Spring", "Autumn"],
            parents=[],
            cpt={
                (): {"Summer": 0.25, "Winter": 0.25, "Spring": 0.25, "Autumn": 0.25}
            }
        )

        # 3. TRANSPORT_TYPE NODE (observed)
        transport_node = BayesianNode(
            name="Transport_Type",
            domain=["Train", "Bus", "Flight"],
            parents=[],
            cpt={
                (): {"Train": 0.4, "Bus": 0.4, "Flight": 0.2}
            }
        )

        # 4. TRANSPORT_DELAYS (dipende da Weather e Transport_Type)
        delays_node = BayesianNode(
            name="Transport_Delays",
            domain=["None", "Minor", "Major"],
            parents=["Weather", "Transport_Type"],
            cpt={}
        )

        # CPT per Transport_Delays (knowledge-based)
        for weather in ["Good", "Fair", "Bad"]:
            for transport in ["Train", "Bus", "Flight"]:
                if weather == "Good":
                    if transport == "Flight":
                        delays_node.cpt[(weather, transport)] = {"None": 0.8, "Minor": 0.15, "Major": 0.05}
                    elif transport == "Train":
                        delays_node.cpt[(weather, transport)] = {"None": 0.85, "Minor": 0.12, "Major": 0.03}
                    else:  # Bus
                        delays_node.cpt[(weather, transport)] = {"None": 0.9, "Minor": 0.08, "Major": 0.02}

                elif weather == "Fair":
                    if transport == "Flight":
                        delays_node.cpt[(weather, transport)] = {"None": 0.6, "Minor": 0.25, "Major": 0.15}
                    elif transport == "Train":
                        delays_node.cpt[(weather, transport)] = {"None": 0.7, "Minor": 0.25, "Major": 0.05}
                    else:  # Bus
                        delays_node.cpt[(weather, transport)] = {"None": 0.8, "Minor": 0.15, "Major": 0.05}

                else:  # Bad weather
                    if transport == "Flight":
                        delays_node.cpt[(weather, transport)] = {"None": 0.3, "Minor": 0.4, "Major": 0.3}
                    elif transport == "Train":
                        delays_node.cpt[(weather, transport)] = {"None": 0.5, "Minor": 0.35, "Major": 0.15}
                    else:  # Bus
                        delays_node.cpt[(weather, transport)] = {"None": 0.6, "Minor": 0.3, "Major": 0.1}

        # 5. TRIP_SUCCESS (dipende da Transport_Delays)
        success_node = BayesianNode(
            name="Trip_Success",
            domain=["Success", "Partial", "Failed"],
            parents=["Transport_Delays"],
            cpt={
                ("None",): {"Success": 0.95, "Partial": 0.04, "Failed": 0.01},
                ("Minor",): {"Success": 0.8, "Partial": 0.15, "Failed": 0.05},
                ("Major",): {"Success": 0.4, "Partial": 0.35, "Failed": 0.25}
            }
        )

        # 6. USER_SATISFACTION (dipende da Trip_Success e costo stimato)
        satisfaction_node = BayesianNode(
            name="User_Satisfaction",
            domain=["High", "Medium", "Low"],
            parents=["Trip_Success"],
            cpt={
                ("Success",): {"High": 0.8, "Medium": 0.15, "Low": 0.05},
                ("Partial",): {"High": 0.3, "Medium": 0.5, "Low": 0.2},
                ("Failed",): {"High": 0.05, "Medium": 0.25, "Low": 0.7}
            }
        )

        # Aggiungi nodi alla rete
        self.nodes = {
            "Weather": weather_node,
            "Season": season_node,
            "Transport_Type": transport_node,
            "Transport_Delays": delays_node,
            "Trip_Success": success_node,
            "User_Satisfaction": satisfaction_node
        }

        print(f"[DONE] Rete Bayesiana creata con {len(self.nodes)} nodi")
        print(f"   Nodi: {list(self.nodes.keys())}")

    def get_probability(self, node_name: str, value: str, evidence: Dict[str, str] = None) -> float:
        """
        Calcola probabilità condizionale P(node=value | evidence)

        ALGORITMO: Enumeration-based inference
        - Semplice ma esatto per reti piccole
        - Marginalizza su variabili non osservate
        """

        if evidence is None:
            evidence = {}

        node = self.nodes[node_name]

        # Se nodo non ha genitori, ritorna probabilità marginale
        if not node.parents:
            return node.cpt[()][value]

        # Calcola probabilità condizionale dato evidence sui genitori
        parent_values = []
        for parent in node.parents:
            if parent in evidence:
                parent_values.append(evidence[parent])
            else:
                # Se genitore non osservato, marginalizza
                return self._marginalize_probability(node_name, value, evidence)

        parent_key = tuple(parent_values) if len(parent_values) > 1 else (parent_values[0],)

        if parent_key in node.cpt:
            return node.cpt[parent_key][value]
        else:
            return 0.0

    def _marginalize_probability(self, node_name: str, value: str, evidence: Dict[str, str]) -> float:
        """
        Marginalizzazione per genitori non osservati
        P(X=x|E) = Σ_y P(X=x|Y=y,E) * P(Y=y|E)
        """

        node = self.nodes[node_name]
        total_prob = 0.0

        # Trova genitori non osservati
        unobserved_parents = [p for p in node.parents if p not in evidence]

        if not unobserved_parents:
            # Tutti genitori osservati
            parent_values = [evidence[p] for p in node.parents]
            parent_key = tuple(parent_values) if len(parent_values) > 1 else (parent_values[0],)
            return node.cpt.get(parent_key, {}).get(value, 0.0)

        # Enumera tutte le combinazioni genitori non osservati
        unobserved_domains = [self.nodes[p].domain for p in unobserved_parents]

        for combo in itertools.product(*unobserved_domains):
            # Costruisci evidence completo
            full_evidence = evidence.copy()
            for i, parent in enumerate(unobserved_parents):
                full_evidence[parent] = combo[i]

            # P(X=x | genitori)
            parent_values = [full_evidence[p] for p in node.parents]
            parent_key = tuple(parent_values) if len(parent_values) > 1 else (parent_values[0],)
            conditional_prob = node.cpt.get(parent_key, {}).get(value, 0.0)

            # P(genitori non osservati | evidence)
            joint_prob = conditional_prob
            for i, parent in enumerate(unobserved_parents):
                parent_prob = self.get_probability(parent, combo[i], evidence)
                joint_prob *= parent_prob

            total_prob += joint_prob

        return total_prob

    def predict_trip_outcome(self,
                           transport_type: str,
                           weather_condition: str = None,
                           season: str = None) -> Dict[str, float]:
        """
        Predice outcome viaggio dato contesto

        APPLICAZIONE PRATICA:
        - Input: tipo trasporto, condizioni meteo
        - Output: probabilità successo/ritardi/soddisfazione
        - Usage: decision support per travel planning
        """

        evidence = {"Transport_Type": transport_type}

        if weather_condition:
            evidence["Weather"] = weather_condition
        if season:
            evidence["Season"] = season

        # Inferenza su variabili di interesse
        results = {}

        # Probabilità ritardi
        delays_probs = {}
        for delay_level in self.nodes["Transport_Delays"].domain:
            delays_probs[delay_level] = self.get_probability("Transport_Delays", delay_level, evidence)
        results["delays"] = delays_probs

        # Probabilità successo viaggio
        success_probs = {}
        for outcome in self.nodes["Trip_Success"].domain:
            success_probs[outcome] = self.get_probability("Trip_Success", outcome, evidence)
        results["trip_success"] = success_probs

        # Probabilità soddisfazione
        satisfaction_probs = {}
        for level in self.nodes["User_Satisfaction"].domain:
            satisfaction_probs[level] = self.get_probability("User_Satisfaction", level, evidence)
        results["user_satisfaction"] = satisfaction_probs

        return results

    def get_best_transport_choice(self,
                                weather_condition: str,
                                utility_weights: Dict[str, float] = None) -> Tuple[str, float]:
        """
        Decision Theory: scelta ottimale trasporto sotto incertezza

        CONCETTI:
        - Expected Utility Theory
        - Decision trees con probabilità
        - Bayesian decision making
        """

        if utility_weights is None:
            utility_weights = {
                "Success": 1.0,
                "Partial": 0.5,
                "Failed": 0.0
            }

        best_transport = None
        best_expected_utility = -1

        transport_results = {}

        for transport in ["Train", "Bus", "Flight"]:
            prediction = self.predict_trip_outcome(transport, weather_condition)

            # Calcola expected utility
            expected_utility = 0.0
            for outcome, prob in prediction["trip_success"].items():
                expected_utility += prob * utility_weights[outcome]

            transport_results[transport] = {
                'expected_utility': expected_utility,
                'predictions': prediction
            }

            if expected_utility > best_expected_utility:
                best_expected_utility = expected_utility
                best_transport = transport

        return best_transport, transport_results

    def monte_carlo_simulation(self,
                             evidence: Dict[str, str],
                             n_samples: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Simulazione Monte Carlo per inferenza approssimata

        CONCETTI:
        - Approximate Inference: quando exact inference troppo costoso
        - Sampling methods: generazione campioni da distribuzione
        - Law of Large Numbers: convergenza a probabilità vera
        """

        print(f"[SIMULATION] Monte Carlo con {n_samples} campioni...")

        # Contatori per ogni variabile
        counts = {}
        for node_name, node in self.nodes.items():
            counts[node_name] = {value: 0 for value in node.domain}

        # Genera campioni
        for _ in range(n_samples):
            sample = self._generate_sample(evidence)

            for node_name, value in sample.items():
                counts[node_name][value] += 1

        # Converti in probabilità
        results = {}
        for node_name, node_counts in counts.items():
            total = sum(node_counts.values())
            results[node_name] = {
                value: count / total for value, count in node_counts.items()
            }

        return results

    def _generate_sample(self, evidence: Dict[str, str]) -> Dict[str, str]:
        """
        Genera singolo campione dalla rete Bayesiana
        Forward sampling rispettando topological order
        """

        sample = evidence.copy()

        # Topological order (genitori prima di figli)
        order = ["Weather", "Season", "Transport_Type", "Transport_Delays", "Trip_Success", "User_Satisfaction"]

        for node_name in order:
            if node_name in sample:
                continue  # Already observed

            node = self.nodes[node_name]

            # Trova valori genitori
            parent_values = []
            for parent in node.parents:
                parent_values.append(sample[parent])

            parent_key = tuple(parent_values) if parent_values else ()

            # Campiona da distribuzione condizionale
            if parent_key in node.cpt:
                probs = node.cpt[parent_key]
                values = list(probs.keys())
                probabilities = list(probs.values())

                sample[node_name] = np.random.choice(values, p=probabilities)
            else:
                # Fallback: uniform sampling
                sample[node_name] = random.choice(node.domain)

        return sample

    def explain_reasoning(self,
                        query_node: str,
                        query_value: str,
                        evidence: Dict[str, str]) -> str:
        """
        Spiega il ragionamento probabilistico in linguaggio naturale

         AI spiegabile per trasparenza
        """

        prob = self.get_probability(query_node, query_value, evidence)

        explanation = f"Probabilità {query_node} = {query_value}: {prob:.3f}\n\n"
        explanation += "Ragionamento:\n"

        if evidence:
            explanation += "Dato che:\n"
            for var, val in evidence.items():
                explanation += f"• {var} = {val}\n"
            explanation += "\n"

        # Trova pathway causale più probabile
        node = self.nodes[query_node]
        if node.parents:
            explanation += f"Fattori che influenzano {query_node}:\n"
            for parent in node.parents:
                if parent in evidence:
                    parent_val = evidence[parent]
                    explanation += f"• {parent} = {parent_val} (osservato)\n"
                else:
                    # Marginalizzazione
                    explanation += f"• {parent} (non osservato, marginalizzato)\n"

        return explanation

# Test della rete Bayesiana
if __name__ == "__main__":

    print("=" * 60)
    print("BAYESIAN NETWORK FOR TRAVEL UNCERTAINTY")
    print("=" * 60)

    # Costruisci rete
    bn = TravelUncertaintyNetwork()

    # Test 1: Probabilità base
    print(f"\n=== TEST 1: PROBABILITÀ BASE ===")
    weather_good = bn.get_probability("Weather", "Good")
    print(f"P(Weather = Good) = {weather_good:.3f}")

    # Test 2: Inferenza condizionale
    print(f"\n=== TEST 2: INFERENZA CONDIZIONALE ===")
    evidence = {"Weather": "Bad", "Transport_Type": "Flight"}
    delay_major = bn.get_probability("Transport_Delays", "Major", evidence)
    print(f"P(Ritardi = Major | Meteo = Bad, Trasporto = Flight) = {delay_major:.3f}")

    # Test 3: Predizione outcome completo
    print(f"\n=== TEST 3: PREDIZIONE TRIP OUTCOME ===")

    test_scenarios = [
        ("Train", "Good", "Summer"),
        ("Flight", "Bad", "Winter"),
        ("Bus", "Fair", "Spring")
    ]

    for transport, weather, season in test_scenarios:
        print(f"\nScenario: {transport} con {weather} weather in {season}")
        prediction = bn.predict_trip_outcome(transport, weather, season)

        print(f"  Ritardi: {', '.join([f'{k}={v:.2f}' for k,v in prediction['delays'].items()])}")
        print(f"  Successo: {', '.join([f'{k}={v:.2f}' for k,v in prediction['trip_success'].items()])}")
        print(f"  Soddisfazione: {', '.join([f'{k}={v:.2f}' for k,v in prediction['user_satisfaction'].items()])}")

    # Test 4: Decision making
    print(f"\n=== TEST 4: BAYESIAN DECISION MAKING ===")

    weather_conditions = ["Good", "Fair", "Bad"]

    for weather in weather_conditions:
        best_transport, results = bn.get_best_transport_choice(weather)
        print(f"\nMeteo {weather}: Miglior scelta = {best_transport}")

        for transport, data in results.items():
            print(f"  {transport}: Expected Utility = {data['expected_utility']:.3f}")

    # Test 5: Monte Carlo simulation
    print(f"\n=== TEST 5: MONTE CARLO SIMULATION ===")

    evidence = {"Weather": "Fair"}
    mc_results = bn.monte_carlo_simulation(evidence, n_samples=5000)

    print(f"Simulazione con {evidence}:")
    for node, probs in mc_results.items():
        if node != "Weather":  # Skip observed
            print(f"  {node}: {', '.join([f'{k}={v:.3f}' for k,v in probs.items()])}")

    # Test 6: Spiegazione reasoning
    print(f"\n=== TEST 6: EXPLAINABLE REASONING ===")

    evidence = {"Weather": "Bad", "Transport_Type": "Flight"}
    explanation = bn.explain_reasoning("Trip_Success", "Failed", evidence)
    print(explanation)

    print(f"\n[DONE] Bayesian Network testing completed!")