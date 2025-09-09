import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from datetime import datetime, time

class CityGraph:
    """
    Grafo delle città italiane per pianificazione viaggi

    CONCETTI:
    - Graph Theory: rappresentazione grafo per algoritmi ricerca
    - NetworkX: libreria per grafi e algoritmi (Dijkstra, A*)
    - Real Data: coordinate GPS reali, non simulate
    """

    def __init__(self):
        # 20 città italiane principali con coordinate GPS reali
        self.cities = {
            'Milano': (45.4642, 9.1900),
            'Roma': (41.9028, 12.4964),
            'Napoli': (40.8518, 14.2681),
            'Torino': (45.0703, 7.6869),
            'Bologna': (44.4949, 11.3426),
            'Firenze': (43.7696, 11.2558),
            'Bari': (41.1177, 16.8719),
            'Palermo': (38.1157, 13.3613),
            'Venezia': (45.4408, 12.3155),
            'Genova': (44.4056, 8.9463),
            'Verona': (45.4384, 10.9916),
            'Catania': (37.5079, 15.0830),
            'Cagliari': (39.2238, 9.1217),
            'Trieste': (45.6495, 13.7768),
            'Perugia': (43.1107, 12.3908),
            'Pescara': (42.4584, 14.2081),
            'Reggio Calabria': (38.1061, 15.6444),
            'Salerno': (40.6824, 14.7681),
            'Brescia': (45.5416, 10.2118),
            'Pisa': (43.7228, 10.4017)
        }

        self.graph = self._build_transport_graph()

    def _build_transport_graph(self) -> nx.Graph:
        """
        Costruisce grafo NetworkX con collegamenti realistici
        Come nel progetto Trip-planner ma per città italiane
        """
        G = nx.Graph()

        # Aggiungi nodi (città) con attributi
        for city, (lat, lon) in self.cities.items():
            G.add_node(city, lat=lat, lon=lon)

        # Collegamenti principali (non tutte le città sono collegate)
        connections = [
            # Nord Italia
            ('Milano', 'Torino', {'train': True, 'bus': True, 'flight': False}),
            ('Milano', 'Venezia', {'train': True, 'bus': True, 'flight': False}),
            ('Milano', 'Bologna', {'train': True, 'bus': True, 'flight': False}),
            ('Milano', 'Genova', {'train': True, 'bus': True, 'flight': False}),
            ('Torino', 'Genova', {'train': True, 'bus': True, 'flight': False}),
            ('Venezia', 'Verona', {'train': True, 'bus': True, 'flight': False}),
            ('Verona', 'Brescia', {'train': True, 'bus': True, 'flight': False}),

            # Centro Italia
            ('Bologna', 'Firenze', {'train': True, 'bus': True, 'flight': False}),
            ('Firenze', 'Roma', {'train': True, 'bus': True, 'flight': False}),
            ('Roma', 'Napoli', {'train': True, 'bus': True, 'flight': False}),
            ('Perugia', 'Roma', {'train': True, 'bus': True, 'flight': False}),
            ('Pisa', 'Firenze', {'train': True, 'bus': True, 'flight': False}),

            # Sud Italia
            ('Napoli', 'Bari', {'train': True, 'bus': True, 'flight': False}),
            ('Napoli', 'Salerno', {'train': True, 'bus': True, 'flight': False}),
            ('Bari', 'Reggio Calabria', {'train': True, 'bus': True, 'flight': False}),

            # Isole e collegamenti lunghi (solo voli)
            ('Roma', 'Palermo', {'train': False, 'bus': False, 'flight': True}),
            ('Milano', 'Palermo', {'train': False, 'bus': False, 'flight': True}),
            ('Roma', 'Cagliari', {'train': False, 'bus': False, 'flight': True}),
            ('Milano', 'Cagliari', {'train': False, 'bus': False, 'flight': True}),
            ('Catania', 'Roma', {'train': False, 'bus': False, 'flight': True}),
        ]

        for city1, city2, transport_modes in connections:
            distance = self._calculate_real_distance(city1, city2)
            G.add_edge(city1, city2, distance=distance, **transport_modes)

        return G

    def _calculate_real_distance(self, city1: str, city2: str) -> float:
        """
        Calcola distanza reale usando formula haversine
        Più accurata della distanza euclidea
        """
        lat1, lon1 = np.radians(self.cities[city1])
        lat2, lon2 = np.radians(self.cities[city2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Raggio Terra in km

        return r * c

    def get_shortest_path(self, origin: str, destination: str, algorithm='dijkstra') -> List[str]:
        """
        Trova percorso più breve utilizzando algoritmo di Dijkstra

        CONCETTI:
        - Dijkstra: algoritmo shortest path classico
        - A*: euristica per ottimizzazione
        - NetworkX: implementazioni ottimizzate
        """
        if algorithm == 'dijkstra':
            try:
                path = nx.shortest_path(self.graph, origin, destination, weight='distance')
                return path
            except nx.NetworkXNoPath:
                return []  # Nessun percorso disponibile

        elif algorithm == 'astar':
            def heuristic(a, b):
                return self._calculate_real_distance(a, b)

            try:
                path = nx.astar_path(self.graph, origin, destination,
                                   heuristic=heuristic, weight='distance')
                return path
            except nx.NetworkXNoPath:
                return []

    def get_path_info(self, path: List[str]) -> Dict:
        """
        Calcola informazioni complete del percorso
        """
        if len(path) < 2:
            return {'total_distance': 0, 'segments': []}

        total_distance = 0
        segments = []

        for i in range(len(path) - 1):
            city1, city2 = path[i], path[i+1]
            edge_data = self.graph[city1][city2]
            distance = edge_data['distance']

            # Trova mezzi disponibili
            available_transport = []
            if edge_data.get('train', False):
                available_transport.append('train')
            if edge_data.get('bus', False):
                available_transport.append('bus')
            if edge_data.get('flight', False):
                available_transport.append('flight')

            segments.append({
                'from': city1,
                'to': city2,
                'distance': round(distance, 2),
                'transport_options': available_transport
            })

            total_distance += distance

        return {
            'total_distance': round(total_distance, 2),
            'segments': segments,
            'cities_count': len(path)
        }

    def get_all_shortest_paths(self) -> Dict:
        """
        Calcola Floyd-Warshall: tutte le distanze minime

         Dynamic Programming
        NetworkX implementa Floyd-Warshall ottimizzato
        """
        # Floyd-Warshall con NetworkX
        all_pairs = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='distance'))

        return all_pairs

    def get_cities_within_radius(self, center_city: str, max_distance: float) -> List[str]:
        """
        Trova città raggiungibili entro distanza massima
        Utile per suggerimenti destinazioni
        """
        reachable = []
        center_pos = self.cities[center_city]

        for city, pos in self.cities.items():
            if city != center_city:
                distance = self._calculate_real_distance(center_city, city)
                if distance <= max_distance:
                    reachable.append((city, distance))

        return sorted(reachable, key=lambda x: x[1])  # Ordina per distanza

# Test del grafo città
if __name__ == "__main__":
    city_graph = CityGraph()

    print("=== GRAFO CITTÀ ITALIANE ===")
    print(f"Città totali: {len(city_graph.cities)}")
    print(f"Collegamenti: {city_graph.graph.number_of_edges()}")

    # Test Dijkstra
    path = city_graph.get_shortest_path('Milano', 'Roma')
    print(f"\n=== PERCORSO Milano -> Roma (Dijkstra) ===")
    print(f"Path: {' -> '.join(path)}")

    # Info dettagliate percorso
    path_info = city_graph.get_path_info(path)
    print(f"Distanza totale: {path_info['total_distance']} km")
    print("Segmenti:")
    for segment in path_info['segments']:
        print(f"  {segment['from']} -> {segment['to']}: {segment['distance']} km")
        print(f"    Trasporti: {segment['transport_options']}")

    # Test A*
    path_astar = city_graph.get_shortest_path('Milano', 'Roma', algorithm='astar')
    print(f"\n=== PERCORSO Milano -> Roma (A*) ===")
    print(f"Path: {' -> '.join(path_astar)}")

    # Città raggiungibili
    nearby = city_graph.get_cities_within_radius('Milano', 300)
    print(f"\n=== CITTÀ RAGGIUNGIBILI DA MILANO (< 300km) ===")
    for city, distance in nearby[:5]:
        print(f"{city}: {distance:.1f} km")