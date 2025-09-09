import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collection.transport_data import CityGraph

class TravelDatasetGenerator:
    """
    Generatore di dataset sintetici per training ML models

    CONCETTI:
    - Feature Engineering: trasformazione dati grafi --> features ML
    - Synthetic Data: simulazione realistica per supervised learning
    - Domain Knowledge: incorpora logica di business nel data generation
    """

    def __init__(self, city_graph: CityGraph):
        self.city_graph = city_graph
        self.cities = list(city_graph.cities.keys())

        # Stagionalità e pattern temporali realistici
        self.seasonal_factors = {
            'winter': {'demand': 0.7, 'price_multiplier': 0.85},
            'spring': {'demand': 1.1, 'price_multiplier': 1.05},
            'summer': {'demand': 1.4, 'price_multiplier': 1.25},
            'autumn': {'demand': 0.9, 'price_multiplier': 0.95}
        }

        # Profili utente con comportamenti diversi
        self.user_profiles = {
            'business': {
                'price_sensitivity': 0.2,
                'time_priority': 0.9,
                'comfort_priority': 0.8,
                'preferred_transports': ['flight', 'train']
            },
            'leisure': {
                'price_sensitivity': 0.8,
                'time_priority': 0.4,
                'comfort_priority': 0.6,
                'preferred_transports': ['bus', 'train']
            },
            'budget': {
                'price_sensitivity': 0.95,
                'time_priority': 0.2,
                'comfort_priority': 0.3,
                'preferred_transports': ['bus']
            }
        }

        # Fattori di domanda per città (tourism attractiveness)
        self.city_demand_factors = {
            'Roma': 1.4, 'Milano': 1.3, 'Venezia': 1.5, 'Firenze': 1.4,
            'Napoli': 1.2, 'Palermo': 1.1, 'Torino': 1.0, 'Bologna': 1.0,
            'Bari': 0.9, 'Genova': 0.9, 'Verona': 1.1, 'Catania': 0.8,
            'Cagliari': 0.8, 'Trieste': 0.7, 'Perugia': 0.8, 'Pescara': 0.6,
            'Reggio Calabria': 0.6, 'Salerno': 0.7, 'Brescia': 0.7, 'Pisa': 0.9
        }

    def generate_travel_scenarios(self, n_scenarios: int = 1000) -> pd.DataFrame:
        """
        Genera dataset completo per training ML models

        Features generate:
        - Geographic: distanza, coordinate lat/lon
        - Temporal: stagione, giorno settimana, ora
        - Transport: tipo mezzo, disponibilità
        - User: profilo, preferenze, storico
        - Economic: domanda, prezzi base
        - Contextual: meteo, eventi speciali
        """

        scenarios = []

        print(f"[DATASET] Generando {n_scenarios} scenari di viaggio...")

        for i in range(n_scenarios):
            # 1. GEOGRAPHIC FEATURES
            origin = random.choice(self.cities)
            destination = random.choice([c for c in self.cities if c != origin])

            # Solo città raggiungibili (controllo grafo)
            try:
                path = self.city_graph.get_shortest_path(origin, destination)
                if not path:
                    continue
            except:
                continue

            distance = self.city_graph._calculate_real_distance(origin, destination)

            # 2. TEMPORAL FEATURES
            season = random.choice(['winter', 'spring', 'summer', 'autumn'])
            day_of_week = random.randint(0, 6)  # 0=Monday, 6=Sunday
            hour = random.randint(6, 22)  # Orari realistici 6:00-22:00

            is_weekend = day_of_week >= 5
            is_peak_hour = hour in [8, 9, 17, 18, 19]

            # 3. USER FEATURES
            user_profile = random.choice(['business', 'leisure', 'budget'])
            user_age = random.randint(18, 70)
            user_income = self._generate_income_by_profile(user_profile)

            # 4. TRANSPORT FEATURES
            available_transports = self._get_available_transports(origin, destination)
            chosen_transport = self._simulate_transport_choice(
                user_profile, available_transports, distance
            )

            # 5. DEMAND & PRICING FEATURES
            base_demand = self._calculate_base_demand(
                origin, destination, season, is_weekend, hour
            )

            # 6. CONTEXTUAL FEATURES
            weather_factor = random.uniform(0.8, 1.2)  # Meteo impact
            special_event = random.choice([0, 0, 0, 1])  # 25% eventi speciali

            # 7. TARGET VARIABLES (quello che vogliamo predire)
            actual_price = self._calculate_realistic_price(
                chosen_transport, distance, base_demand, season,
                is_weekend, weather_factor, special_event
            )

            actual_time = self._calculate_realistic_time(
                chosen_transport, distance, weather_factor, is_peak_hour
            )

            user_satisfaction = self._calculate_satisfaction(
                user_profile, chosen_transport, actual_price, actual_time
            )

            # 8. ASSEMBLY SCENARIO
            scenario = {
                # Geographic
                'origin': origin,
                'destination': destination,
                'distance': distance,
                'origin_lat': self.city_graph.cities[origin][0],
                'origin_lon': self.city_graph.cities[origin][1],
                'dest_lat': self.city_graph.cities[destination][0],
                'dest_lon': self.city_graph.cities[destination][1],

                # Temporal
                'season': season,
                'day_of_week': day_of_week,
                'hour': hour,
                'is_weekend': is_weekend,
                'is_peak_hour': is_peak_hour,

                # User
                'user_profile': user_profile,
                'user_age': user_age,
                'user_income': user_income,
                'price_sensitivity': self.user_profiles[user_profile]['price_sensitivity'],
                'time_priority': self.user_profiles[user_profile]['time_priority'],
                'comfort_priority': self.user_profiles[user_profile]['comfort_priority'],

                # Transport
                'transport_type': chosen_transport,
                'available_train': 'train' in available_transports,
                'available_bus': 'bus' in available_transports,
                'available_flight': 'flight' in available_transports,

                # Economic
                'base_demand': base_demand,
                'seasonal_multiplier': self.seasonal_factors[season]['price_multiplier'],

                # Contextual
                'weather_factor': weather_factor,
                'special_event': special_event,

                # Targets (da predire)
                'actual_price': actual_price,
                'actual_time': actual_time,
                'user_satisfaction': user_satisfaction
            }

            scenarios.append(scenario)

            if (i + 1) % 100 == 0:
                print(f"   Generati {i + 1}/{n_scenarios} scenari...")

        df = pd.DataFrame(scenarios)
        print(f"[DONE] Dataset generato: {len(df)} scenari con {len(df.columns)} features")

        return df

    def _get_available_transports(self, origin: str, destination: str) -> List[str]:
        """Recupera trasporti disponibili dal grafo"""
        try:
            edge_data = self.city_graph.graph[origin][destination]
            transports = []
            if edge_data.get('train', False):
                transports.append('train')
            if edge_data.get('bus', False):
                transports.append('bus')
            if edge_data.get('flight', False):
                transports.append('flight')
            return transports
        except:
            return ['bus']  # Fallback

    def _simulate_transport_choice(self,
                                 user_profile: str,
                                 available_transports: List[str],
                                 distance: float) -> str:
        """Simula scelta trasporto basata su profilo utente e distanza"""
        preferences = self.user_profiles[user_profile]['preferred_transports']

        # Filtra trasporti disponibili che piacciono all'utente
        preferred_available = [t for t in preferences if t in available_transports]

        if preferred_available:
            # Logica distanza: voli per >800km, treni per 100-800km, bus per <300km
            if distance > 800 and 'flight' in preferred_available:
                return 'flight'
            elif distance > 100 and 'train' in preferred_available:
                return 'train'
            else:
                return random.choice(preferred_available)
        else:
            # Fallback su disponibili
            return random.choice(available_transports) if available_transports else 'bus'

    def _generate_income_by_profile(self, profile: str) -> float:
        """Genera reddito realistico per profilo"""
        income_ranges = {
            'business': (40000, 80000),
            'leisure': (25000, 50000),
            'budget': (15000, 30000)
        }
        low, high = income_ranges[profile]
        return random.uniform(low, high)

    def _calculate_base_demand(self, origin: str, destination: str,
                             season: str, is_weekend: bool, hour: int) -> float:
        """Calcola domanda base per pricing"""

        # Fattore città destinazione (turismo)
        dest_factor = self.city_demand_factors.get(destination, 1.0)

        # Fattore stagionale
        season_factor = self.seasonal_factors[season]['demand']

        # Fattore weekend
        weekend_factor = 1.3 if is_weekend else 1.0

        # Fattore orario (rush hours)
        hour_factor = 1.2 if hour in [8, 9, 17, 18, 19] else 1.0

        base_demand = dest_factor * season_factor * weekend_factor * hour_factor

        # Aggiungi rumore realistico
        noise = random.uniform(0.9, 1.1)

        return base_demand * noise

    def _calculate_realistic_price(self, transport: str, distance: float,
                                 demand: float, season: str, is_weekend: bool,
                                 weather_factor: float, special_event: int) -> float:
        """Calcola prezzo realistico con tutti i fattori"""

        # Prezzi base per km (da transport_data.py)
        base_prices = {
            'train': 0.15,
            'bus': 0.08,
            'flight': 0.25
        }

        base_price = distance * base_prices.get(transport, 0.10)

        # Fattori moltiplicativi
        demand_multiplier = 1.0 + (demand - 1.0) * 0.5  # Demand impact
        seasonal_multiplier = self.seasonal_factors[season]['price_multiplier']
        weekend_multiplier = 1.15 if is_weekend else 1.0
        weather_multiplier = weather_factor
        event_multiplier = 1.3 if special_event else 1.0

        final_price = (base_price * demand_multiplier * seasonal_multiplier *
                      weekend_multiplier * weather_multiplier * event_multiplier)

        # Rumore realistico ±10%
        noise = random.uniform(0.9, 1.1)

        return round(final_price * noise, 2)

    def _calculate_realistic_time(self, transport: str, distance: float,
                                weather_factor: float, is_peak_hour: bool) -> float:
        """Calcola tempo realistico di viaggio"""

        # Velocità base (da transport_data.py)
        base_speeds = {
            'train': 120,
            'bus': 80,
            'flight': 500
        }

        base_time = distance / base_speeds.get(transport, 60)

        # Fattori di ritardo
        weather_delay = weather_factor if weather_factor > 1.0 else 1.0
        peak_delay = 1.2 if is_peak_hour else 1.0

        # Tempi fissi (check-in per voli, etc.)
        fixed_times = {
            'train': 0.5,  # 30 min
            'bus': 0.2,    # 12 min
            'flight': 2.0  # 2 ore (check-in + attesa)
        }

        total_time = (base_time * weather_delay * peak_delay +
                     fixed_times.get(transport, 0))

        # Rumore realistico ±15%
        noise = random.uniform(0.85, 1.15)

        return round(total_time * noise, 2)

    def _calculate_satisfaction(self, user_profile: str, transport: str,
                              price: float, time: float) -> float:
        """Calcola soddisfazione utente (0-1) per training"""

        profile_data = self.user_profiles[user_profile]

        # Normalizza price e time per scoring
        normalized_price = min(price / 200, 1.0)  # Max €200 --> 1.0
        normalized_time = min(time / 10, 1.0)     # Max 10h --> 1.0

        # Score pesato per profilo
        price_penalty = profile_data['price_sensitivity'] * normalized_price
        time_penalty = profile_data['time_priority'] * normalized_time

        # Transport comfort bonus
        transport_comfort = {'flight': 0.6, 'train': 0.8, 'bus': 0.4}
        comfort_bonus = (profile_data['comfort_priority'] *
                        transport_comfort.get(transport, 0.5))

        # Formula soddisfazione
        satisfaction = 1.0 - price_penalty - time_penalty + comfort_bonus * 0.3

        # Clamp 0-1 con rumore
        satisfaction = max(0.1, min(0.9, satisfaction))
        noise = random.uniform(-0.05, 0.05)

        return round(satisfaction + noise, 3)

# Test del generatore
if __name__ == "__main__":
    city_graph = CityGraph()
    generator = TravelDatasetGenerator(city_graph)

    print("=== TRAVEL DATASET GENERATOR ===")

    # Genera dataset di test
    df = generator.generate_travel_scenarios(n_scenarios=200)

    print(f"\n=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\n=== SAMPLE DATA ===")
    print(df[['origin', 'destination', 'transport_type',
              'actual_price', 'actual_time', 'user_satisfaction']].head())

    print(f"\n=== STATISTICS ===")
    print(f"Average price: €{df['actual_price'].mean():.2f}")
    print(f"Average time: {df['actual_time'].mean():.2f}h")
    print(f"Average satisfaction: {df['user_satisfaction'].mean():.3f}")

    # Salva dataset
    output_path = "../data/travel_scenarios.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[SAVED] Dataset salvato in: {output_path}")