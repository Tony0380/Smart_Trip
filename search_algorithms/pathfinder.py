import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import heapq
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.transport_data import CityGraph

class OptimizationObjective(Enum):
    """Obiettivi di ottimizzazione per ricerca multi-criterio"""
    DISTANCE = "distance"
    TIME = "time"
    COST = "cost"
    COMFORT = "comfort"

@dataclass
class RouteSegment:
    """Segmento di viaggio con attributi multi-dimensionali"""
    origin: str
    destination: str
    transport_type: str
    distance: float
    time: float
    cost: float
    comfort: float
    departure_time: int

@dataclass
class TravelRoute:
    """Percorso completo multi-obiettivo"""
    segments: List[RouteSegment]
    total_distance: float
    total_time: float
    total_cost: float
    avg_comfort: float
    path: List[str]

    def __post_init__(self):
        """Calcola score composito per ranking"""
        # Normalizza metriche per comparazione
        self.normalized_score = (
            (1.0 / (1.0 + self.total_distance / 1000)) * 0.3 +  # Distanza
            (1.0 / (1.0 + self.total_time / 10)) * 0.3 +        # Tempo
            (1.0 / (1.0 + self.total_cost / 100)) * 0.2 +       # Costo
            self.avg_comfort * 0.2                               # Comfort
        )

class AdvancedPathfinder:
    """
    Algoritmi di ricerca avanzati per pianificazione viaggi

    ARGOMENTI DEL PROGRAMMA IMPLEMENTATI:
    1. Floyd-Warshall: Dynamic Programming per tutte le distanze
    2. Multi-objective A*: Ottimizzazione multi-criterio
    3. Beam Search: Pruning euristico per performance
    4. Heuristic Functions: Euristiche ammissibili
    """

    def __init__(self, city_graph: CityGraph):
        self.city_graph = city_graph
        self.graph = city_graph.graph
        self.cities = list(city_graph.cities.keys())

        # Trasporti con caratteristiche realistiche
        self.transport_profiles = {
            'train': {'speed_kmh': 120, 'cost_per_km': 0.15, 'comfort': 0.8},
            'bus': {'speed_kmh': 80, 'cost_per_km': 0.08, 'comfort': 0.4},
            'flight': {'speed_kmh': 500, 'cost_per_km': 0.25, 'comfort': 0.6}
        }

        # Pre-calcola Floyd-Warshall
        self.all_distances = self._compute_floyd_warshall()

    def _compute_floyd_warshall(self) -> Dict[str, Dict[str, float]]:
        """
        ALGORITMO 1: Floyd-Warshall

         Dynamic Programming
        - Complessità O(n³) ma calcola TUTTE le distanze minime
        - Utile quando serve interrogare molte coppie origine-destinazione
        - Classico algoritmo di programmazione dinamica
        """
        print("[INFO] Computando Floyd-Warshall per tutte le distanze...")

        # Inizializza matrici distanze
        distances = {}
        for city1 in self.cities:
            distances[city1] = {}
            for city2 in self.cities:
                if city1 == city2:
                    distances[city1][city2] = 0
                elif self.graph.has_edge(city1, city2):
                    distances[city1][city2] = self.graph[city1][city2]['distance']
                else:
                    distances[city1][city2] = float('inf')

        # Floyd-Warshall: provare tutti i nodi intermedi
        for k in self.cities:
            for i in self.cities:
                for j in self.cities:
                    if distances[i][k] + distances[k][j] < distances[i][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]

        print("[DONE] Floyd-Warshall completato!")
        return distances

    def get_floyd_warshall_path(self, origin: str, destination: str) -> List[str]:
        """
        Ricostruisce percorso da Floyd-Warshall
        Nota: Floyd-Warshall calcola distanze, serve algoritmo separato per path
        """
        try:
            # Usa Dijkstra per path reconstruction (Floyd-Warshall è per distanze)
            path = nx.shortest_path(self.graph, origin, destination, weight='distance')
            return path
        except nx.NetworkXNoPath:
            return []

    def multi_objective_astar(self,
                             origin: str,
                             destination: str,
                             objectives: List[OptimizationObjective] = None,
                             weights: List[float] = None) -> Optional[TravelRoute]:
        """
        ALGORITMO 2: Multi-objective A*

         Heuristic Search con multiple objectives
        - A* classico ottimizza 1 criterio, qui ottimizziamo N criteri
        - Weighted sum approach per combinare obiettivi
        - Euristica ammissibile per garantire ottimalità
        """
        if objectives is None:
            objectives = [OptimizationObjective.DISTANCE, OptimizationObjective.TIME]
        if weights is None:
            weights = [1.0] * len(objectives)

        print(f"[A*] Multi-objective A*: {origin} -> {destination}")
        print(f"   Obiettivi: {[obj.value for obj in objectives]}")

        # Priority queue: (f_score, g_score, current_city, path, route_data)
        open_set = [(0, 0, origin, [origin], [])]
        closed_set = set()

        while open_set:
            f_score, g_score, current, path, route_segments = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)

            # Goal raggiunto?
            if current == destination:
                return self._build_travel_route(route_segments, path)

            # Esplora vicini
            for neighbor in self.graph.neighbors(current):
                if neighbor in closed_set:
                    continue

                # Calcola costi multi-obiettivo per questo segmento
                edge_data = self.graph[current][neighbor]
                best_segment = self._find_best_transport_option(current, neighbor, edge_data)

                if best_segment is None:
                    continue

                # Nuovo g_score (costo fino a neighbor)
                new_route_segments = route_segments + [best_segment]
                new_g_score = self._calculate_multi_objective_cost(new_route_segments, objectives, weights)

                # Euristica h_score (stima costo da neighbor a destination)
                h_score = self._multi_objective_heuristic(neighbor, destination, objectives, weights)

                # f_score = g_score + h_score
                new_f_score = new_g_score + h_score
                new_path = path + [neighbor]

                heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, new_path, new_route_segments))

        return None  # Nessun percorso trovato

    def _find_best_transport_option(self, city1: str, city2: str, edge_data: dict) -> Optional[RouteSegment]:
        """
        Trova il trasporto migliore per un segmento
        """
        distance = edge_data['distance']
        available_transports = []

        # Controlla trasporti disponibili
        for transport in ['train', 'bus', 'flight']:
            if edge_data.get(transport, False):
                profile = self.transport_profiles[transport]

                # Calcola attributi del segmento
                time = distance / profile['speed_kmh']
                cost = distance * profile['cost_per_km']
                comfort = profile['comfort']

                available_transports.append(RouteSegment(
                    origin=city1,
                    destination=city2,
                    transport_type=transport,
                    distance=distance,
                    time=time,
                    cost=cost,
                    comfort=comfort,
                    departure_time=9  # Simplified - ore 9:00
                ))

        # Restituisci il trasporto con miglior score medio
        if available_transports:
            return max(available_transports,
                      key=lambda t: (1.0/t.time) + (1.0/t.cost) + t.comfort)
        return None

    def _calculate_multi_objective_cost(self,
                                      segments: List[RouteSegment],
                                      objectives: List[OptimizationObjective],
                                      weights: List[float]) -> float:
        """
        Calcola costo multi-obiettivo weighted sum
        """
        if not segments:
            return 0

        total_distance = sum(s.distance for s in segments)
        total_time = sum(s.time for s in segments)
        total_cost = sum(s.cost for s in segments)
        avg_comfort = sum(s.comfort for s in segments) / len(segments)

        # Mappa obiettivi a valori
        objective_values = {
            OptimizationObjective.DISTANCE: total_distance,
            OptimizationObjective.TIME: total_time,
            OptimizationObjective.COST: total_cost,
            OptimizationObjective.COMFORT: 1.0 / avg_comfort  # Inverti comfort (vogliamo massimizzarlo)
        }

        # Weighted sum
        weighted_cost = 0
        for obj, weight in zip(objectives, weights):
            weighted_cost += weight * objective_values[obj]

        return weighted_cost

    def _multi_objective_heuristic(self,
                                  current: str,
                                  destination: str,
                                  objectives: List[OptimizationObjective],
                                  weights: List[float]) -> float:
        """
        EURISTICA MULTI-OBIETTIVO AMMISSIBILE

         Heuristic Functions
        - Deve essere ammissibile (mai sovrastimare il costo reale)
        - Distanza euclidea come lower bound per tutti gli obiettivi
        - Mantiene ottimalità di A*
        """
        # Distanza euclidea come lower bound
        euclidean_dist = self.city_graph._calculate_real_distance(current, destination)

        # Stima lower bound per ogni obiettivo
        heuristic_values = {
            OptimizationObjective.DISTANCE: euclidean_dist,
            OptimizationObjective.TIME: euclidean_dist / 500,  # Velocità massima (volo)
            OptimizationObjective.COST: euclidean_dist * 0.08,  # Costo minimo (bus)
            OptimizationObjective.COMFORT: euclidean_dist * 0.1  # Placeholder comfort penalty
        }

        # Weighted sum delle euristiche
        weighted_heuristic = 0
        for obj, weight in zip(objectives, weights):
            weighted_heuristic += weight * heuristic_values[obj]

        return weighted_heuristic

    def _build_travel_route(self, segments: List[RouteSegment], path: List[str]) -> TravelRoute:
        """
        Costruisce oggetto TravelRoute finale
        """
        if not segments:
            return None

        total_distance = sum(s.distance for s in segments)
        total_time = sum(s.time for s in segments)
        total_cost = sum(s.cost for s in segments)
        avg_comfort = sum(s.comfort for s in segments) / len(segments)

        return TravelRoute(
            segments=segments,
            total_distance=total_distance,
            total_time=total_time,
            total_cost=total_cost,
            avg_comfort=avg_comfort,
            path=path
        )

    def beam_search_routes(self,
                          origin: str,
                          destination: str,
                          beam_width: int = 3,
                          max_depth: int = 5) -> List[TravelRoute]:
        """
        ALGORITMO 3: Beam Search

         Heuristic Search con Pruning
        - Mantiene solo i migliori K candidati ad ogni livello
        - Trade-off completezza vs efficienza
        - Utile per spazi stati enormi
        """
        print(f"[BEAM] Beam Search: {origin} -> {destination} (beam_width={beam_width})")

        # Stato: (current_city, path, route_segments, total_score)
        current_beam = [(origin, [origin], [], 0.0)]
        completed_routes = []

        for depth in range(max_depth):
            if not current_beam:
                break

            next_beam = []

            for current_city, path, route_segments, score in current_beam:

                # Goal raggiunto?
                if current_city == destination:
                    route = self._build_travel_route(route_segments, path)
                    if route:
                        completed_routes.append(route)
                    continue

                # Esplora vicini
                for neighbor in self.graph.neighbors(current_city):
                    if neighbor in path:  # Evita cicli
                        continue

                    edge_data = self.graph[current_city][neighbor]
                    best_segment = self._find_best_transport_option(current_city, neighbor, edge_data)

                    if best_segment is None:
                        continue

                    new_path = path + [neighbor]
                    new_segments = route_segments + [best_segment]

                    # Score = costo attuale + euristica
                    current_cost = sum(s.cost + s.time for s in new_segments)
                    heuristic = self.city_graph._calculate_real_distance(neighbor, destination)
                    new_score = current_cost + heuristic

                    next_beam.append((neighbor, new_path, new_segments, new_score))

            # PRUNING: mantieni solo i migliori beam_width candidati
            next_beam.sort(key=lambda x: x[3])  # Ordina per score
            current_beam = next_beam[:beam_width]  # Taglia beam

            print(f"  Depth {depth+1}: {len(current_beam)} candidati in beam")

        # Ordina routes completate per score
        completed_routes.sort(key=lambda r: r.normalized_score, reverse=True)
        return completed_routes[:beam_width]

# Test degli algoritmi
if __name__ == "__main__":
    city_graph = CityGraph()
    pathfinder = AdvancedPathfinder(city_graph)

    origin, destination = "Milano", "Roma"

    print("=" * 50)
    print("ADVANCED PATHFINDING ALGORITHMS")
    print("=" * 50)

    # Test 1: Floyd-Warshall
    print(f"\n[1] FLOYD-WARSHALL DISTANCE")
    fw_distance = pathfinder.all_distances[origin][destination]
    fw_path = pathfinder.get_floyd_warshall_path(origin, destination)
    print(f"   Distanza minima: {fw_distance:.2f} km")
    print(f"   Path: {' -> '.join(fw_path)}")

    # Test 2: Multi-objective A*
    print(f"\n[2] MULTI-OBJECTIVE A*")
    objectives = [OptimizationObjective.DISTANCE, OptimizationObjective.TIME, OptimizationObjective.COST]
    weights = [0.4, 0.4, 0.2]

    mo_route = pathfinder.multi_objective_astar(origin, destination, objectives, weights)
    if mo_route:
        print(f"   Path: {' -> '.join(mo_route.path)}")
        print(f"   Distanza: {mo_route.total_distance:.2f} km")
        print(f"   Tempo: {mo_route.total_time:.2f} ore")
        print(f"   Costo: €{mo_route.total_cost:.2f}")
        print(f"   Comfort: {mo_route.avg_comfort:.2f}/1.0")
        print(f"   Score: {mo_route.normalized_score:.3f}")

    # Test 3: Beam Search
    print(f"\n[3] BEAM SEARCH (Top 3 Routes)")
    beam_routes = pathfinder.beam_search_routes(origin, destination, beam_width=3)

    for i, route in enumerate(beam_routes, 1):
        print(f"   Rotta {i}: {' -> '.join(route.path)}")
        print(f"      Distanza: {route.total_distance:.1f}km, "
              f"Tempo: {route.total_time:.1f}h, "
              f"Costo: €{route.total_cost:.0f}, "
              f"Score: {route.normalized_score:.3f}")

    print(f"\n[DONE] Test algoritmi completato!")