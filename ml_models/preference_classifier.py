import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, List, Tuple, Optional
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class UserPreferenceClassifier:
    """
    Classificatore ML per profili utente e preferenze di viaggio

    ARGOMENTI DEL PROGRAMMA IMPLEMENTATI:
    1. Supervised Classification: predire categorie da features
    2. Multi-class Problem: business/leisure/budget profiles
    3. Feature Selection: identificare patterns comportamentali
    4. Model Ensemble: confronto algoritmi diversi
    5. Personalization: adattare raccomandazioni a utente
    """

    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }

        self.param_grids = {
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.is_trained = False
        self.grid_search_results = {}

        # Profili utente con caratteristiche distintive
        self.profile_characteristics = {
            'business': {
                'typical_income': (40000, 80000),
                'price_sensitivity': (0.1, 0.3),
                'time_priority': (0.8, 1.0),
                'comfort_priority': (0.7, 0.9),
                'preferred_transports': ['flight', 'train']
            },
            'leisure': {
                'typical_income': (25000, 50000),
                'price_sensitivity': (0.6, 0.9),
                'time_priority': (0.2, 0.6),
                'comfort_priority': (0.5, 0.8),
                'preferred_transports': ['train', 'bus']
            },
            'budget': {
                'typical_income': (15000, 30000),
                'price_sensitivity': (0.8, 1.0),
                'time_priority': (0.1, 0.4),
                'comfort_priority': (0.2, 0.5),
                'preferred_transports': ['bus']
            }
        }

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Feature Engineering per classificazione profili utente

        CONCETTI:
        - Behavioral Features: estrazione pattern comportamentali
        - Domain Knowledge: uso conoscenza dominio per feature selection
        - Categorical Encoding: gestione variabili categoriche
        """

        print("Preparazione features classificazione utenti")

        # Target: user_profile (business/leisure/budget)
        target = df['user_profile'].copy()

        # 1. USER DEMOGRAPHIC FEATURES
        features = pd.DataFrame({
            'user_age': df['user_age'],
            'user_income': df['user_income'],
            'price_sensitivity': df['price_sensitivity'],
            'time_priority': df['time_priority'],
            'comfort_priority': df['comfort_priority']
        })

        # 2. TRAVEL BEHAVIOR FEATURES
        # Analisi pattern di scelta trasporto
        features['chosen_expensive_transport'] = (df['transport_type'] == 'flight').astype(int)
        features['chosen_fast_transport'] = (df['transport_type'].isin(['flight', 'train'])).astype(int)
        features['chosen_cheap_transport'] = (df['transport_type'] == 'bus').astype(int)

        # 3. TRAVEL CONTEXT FEATURES
        features['travels_weekend'] = df['is_weekend'].astype(int)
        features['travels_peak_hour'] = df['is_peak_hour'].astype(int)
        features['long_distance_trip'] = (df['distance'] > 500).astype(int)
        features['premium_season'] = (df['season'] == 'summer').astype(int)

        # 4. ECONOMIC BEHAVIOR FEATURES
        # Rapporto price/income come indicator di spending behavior
        features['price_income_ratio'] = df['actual_price'] / (df['user_income'] / 12)  # Monthly income
        features['willing_pay_premium'] = (features['price_income_ratio'] > 0.1).astype(int)

        # 5. SATISFACTION PATTERN FEATURES
        features['high_satisfaction'] = (df['user_satisfaction'] > 0.7).astype(int)
        features['low_satisfaction'] = (df['user_satisfaction'] < 0.4).astype(int)

        # 6. DERIVED FEATURES
        # Speed preference (tempo vs prezzo)
        features['speed_over_price'] = (features['time_priority'] > features['price_sensitivity']).astype(int)
        features['comfort_seeker'] = (features['comfort_priority'] > 0.7).astype(int)

        self.feature_names = list(features.columns)

        print(f"Features: {len(self.feature_names)}")

        return features, target

    def train_models_with_grid_search(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """
        Training con Grid Search e valutazione classificatori multi-class

        CONCETTI:
        - Grid Search: ottimizzazione sistematica iperparametri
        - Classification Metrics: accuracy, precision, recall, F1
        - Cross-Validation: valutazione robusta K-fold
        - Model Selection: scelta miglior algoritmo per dominio
        """

        print("Training classificatori profili utente")

        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        results = {}

        for model_name, model in self.models.items():
            print(f"Training {model_name}...")

            # Scaling per modelli che lo richiedono
            if model_name in ['logistic_regression', 'svm']:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            # Grid Search
            if model_name in self.param_grids:
                print(f"Grid search in corso...")
                grid_search = GridSearchCV(
                    model, self.param_grids[model_name],
                    cv=3, scoring='accuracy', n_jobs=1, verbose=0
                )
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                
                self.grid_search_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'grid_scores': grid_search.cv_results_
                }
                print(f"Migliori parametri: {grid_search.best_params_}")
            else:
                best_model = model
                best_model.fit(X_train_scaled, y_train)

            # Cross-validation finale con best model
            cv_scores = cross_val_score(best_model, X_train_scaled, y_train,
                                      cv=5, scoring='accuracy')

            # Test set evaluation
            y_pred = best_model.predict(X_test_scaled)

            # Metriche dettagliate
            test_accuracy = accuracy_score(y_test, y_pred)

            # Classification report per ogni classe
            class_report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )

            results[model_name] = {
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'classification_report': class_report,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'model': best_model
            }

            print(f"Accuracy: {cv_scores.mean():.3f}")

        # Seleziona miglior modello
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        self.is_trained = True

        print(f"Miglior modello: {best_model_name} (Accuracy: {results[best_model_name]['test_accuracy']:.3f})")

        return results

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """
        Wrapper per backward compatibility
        """
        return self.train_models_with_grid_search(X, y)
    
    def get_grid_search_summary(self) -> Dict:
        """
        Restituisce summary dei risultati grid search
        """
        summary = {}
        for model_name, results in self.grid_search_results.items():
            summary[model_name] = {
                'best_params': results['best_params'],
                'best_cv_score': results['best_score']
            }
        return summary

    def predict_user_profile(self,
                           user_age: int,
                           user_income: float,
                           price_sensitivity: float,
                           time_priority: float,
                           comfort_priority: float,
                           **behavioral_features) -> Tuple[str, Dict[str, float]]:
        """
        Predice profilo utente con probabilità

        INTEGRAZIONE con sistema di ricerca:
        - Determina preferenze per pesare obiettivi in A*
        - Personalizza euristiche di ricerca
        - Filtra opzioni trasporto per profilo
        """

        if not self.is_trained:
            # Fallback: classifica per income range
            if user_income > 50000:
                return 'business', {'business': 0.8, 'leisure': 0.15, 'budget': 0.05}
            elif user_income > 30000:
                return 'leisure', {'business': 0.2, 'leisure': 0.7, 'budget': 0.1}
            else:
                return 'budget', {'business': 0.1, 'leisure': 0.3, 'budget': 0.6}

        # Costruisci feature vector
        features = self._build_feature_vector(
            user_age=user_age,
            user_income=user_income,
            price_sensitivity=price_sensitivity,
            time_priority=time_priority,
            comfort_priority=comfort_priority,
            **behavioral_features
        )

        # Scaling se necessario
        if self.best_model_name in ['logistic_regression', 'svm']:
            features_scaled = self.scaler.transform([features])
        else:
            features_scaled = [features]

        # Predizione con probabilità
        predicted_class_idx = self.best_model.predict(features_scaled)[0]
        class_probabilities = self.best_model.predict_proba(features_scaled)[0]

        predicted_profile = self.label_encoder.inverse_transform([predicted_class_idx])[0]

        # Mappa probabilità a nomi classi
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = class_probabilities[i]

        return predicted_profile, prob_dict

    def _build_feature_vector(self, **params) -> List[float]:
        """Costruisce feature vector per predizione singola"""

        # Feature base fornite
        base_features = [
            params.get('user_age', 35),
            params.get('user_income', 35000),
            params.get('price_sensitivity', 0.6),
            params.get('time_priority', 0.5),
            params.get('comfort_priority', 0.5)
        ]

        # Behavioral features (default o calcolate)
        behavioral = [
            params.get('chosen_expensive_transport', 0),
            params.get('chosen_fast_transport', 1),
            params.get('chosen_cheap_transport', 0),
            params.get('travels_weekend', 0),
            params.get('travels_peak_hour', 0),
            params.get('long_distance_trip', 1),
            params.get('premium_season', 1),
            params.get('price_income_ratio', 0.05),
            params.get('willing_pay_premium', 0),
            params.get('high_satisfaction', 1),
            params.get('low_satisfaction', 0),
            # Derived features
            int(params.get('time_priority', 0.5) > params.get('price_sensitivity', 0.6)),  # speed_over_price
            int(params.get('comfort_priority', 0.5) > 0.7)  # comfort_seeker
        ]

        return base_features + behavioral

    def get_personalized_weights(self, user_profile: str, probabilities: Dict[str, float] = None) -> Dict[str, float]:
        """
        Ottieni pesi personalizzati per algoritmi di ricerca multi-obiettivo

        INTEGRAZIONE:
        - Pesi per A* multi-obiettivo basati su profilo predetto
        - Personalizzazione euristica di ricerca
        - Bilanciamento automatico costo/tempo/comfort
        """

        # Pesi base per profilo
        base_weights = {
            'business': {
                'distance': 0.15,    # Meno importante
                'time': 0.45,        # Molto importante
                'cost': 0.20,        # Moderatamente importante
                'comfort': 0.20      # Importante
            },
            'leisure': {
                'distance': 0.25,    # Importante per esplorazione
                'time': 0.25,        # Moderato
                'cost': 0.35,        # Importante
                'comfort': 0.15      # Meno importante
            },
            'budget': {
                'distance': 0.20,    # Moderato
                'time': 0.15,        # Meno importante
                'cost': 0.50,        # Priorità assoluta
                'comfort': 0.15      # Meno importante
            }
        }

        # Se abbiamo probabilità, facciamo weighted average
        if probabilities:
            final_weights = {'distance': 0, 'time': 0, 'cost': 0, 'comfort': 0}

            for profile, prob in probabilities.items():
                if profile in base_weights:
                    for objective, weight in base_weights[profile].items():
                        final_weights[objective] += prob * weight

            return final_weights
        else:
            return base_weights.get(user_profile, base_weights['leisure'])

    def explain_prediction(self, user_profile: str, probabilities: Dict[str, float]) -> str:
        """
        Genera spiegazione human-readable della classificazione

         AI spiegabile per trasparenza sistema
        """

        explanation = f"Profilo predetto: {user_profile.upper()}\n"
        explanation += f"Confidenza: {probabilities[user_profile]:.1%}\n\n"

        explanation += "Caratteristiche del profilo:\n"
        characteristics = self.profile_characteristics[user_profile]

        explanation += f"• Reddito tipico: €{characteristics['typical_income'][0]:,}-{characteristics['typical_income'][1]:,}\n"
        explanation += f"• Sensibilità prezzo: {characteristics['price_sensitivity'][0]:.1f}-{characteristics['price_sensitivity'][1]:.1f}\n"
        explanation += f"• Priorità tempo: {characteristics['time_priority'][0]:.1f}-{characteristics['time_priority'][1]:.1f}\n"
        explanation += f"• Trasporti preferiti: {', '.join(characteristics['preferred_transports'])}\n"

        return explanation

    def save_model(self, filepath: str):
        """Salva classificatore trainato"""
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'profile_characteristics': self.profile_characteristics
        }
        joblib.dump(model_data, filepath)
        print(f"Classificatore salvato: {filepath}")

    def load_model(self, filepath: str):
        """Carica classificatore pre-trainato"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.profile_characteristics = model_data['profile_characteristics']
        self.is_trained = True
        print(f"Classificatore caricato: {filepath}")

# Test del classificatore
if __name__ == "__main__":
    from dataset_generator import TravelDatasetGenerator, CityGraph

    print("=== USER PREFERENCE CLASSIFIER TRAINING ===")

    # Genera dataset
    city_graph = CityGraph()
    generator = TravelDatasetGenerator(city_graph)
    df = generator.generate_travel_scenarios(n_scenarios=800)

    # Train classificatore
    classifier = UserPreferenceClassifier()

    X, y = classifier.prepare_features(df)
    results = classifier.train_models(X, y)

    # Risultati dettagliati
    print(f"\n=== CLASSIFICATION RESULTS ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"   CV Accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
        print(f"   Test Accuracy: {metrics['test_accuracy']:.3f}")

        # Per ogni classe
        for class_name in ['business', 'leisure', 'budget']:
            if class_name in metrics['classification_report']:
                class_metrics = metrics['classification_report'][class_name]
                print(f"   {class_name}: P={class_metrics['precision']:.3f}, R={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}")

    # Test predizioni
    print(f"\n=== PREDICTION TESTS ===")

    test_users = [
        {'user_age': 35, 'user_income': 60000, 'price_sensitivity': 0.2,
         'time_priority': 0.9, 'comfort_priority': 0.8},
        {'user_age': 28, 'user_income': 32000, 'price_sensitivity': 0.75,
         'time_priority': 0.4, 'comfort_priority': 0.6},
        {'user_age': 22, 'user_income': 20000, 'price_sensitivity': 0.95,
         'time_priority': 0.2, 'comfort_priority': 0.3}
    ]

    for i, user in enumerate(test_users, 1):
        profile, probs = classifier.predict_user_profile(**user)
        weights = classifier.get_personalized_weights(profile, probs)

        print(f"\nUser {i}: Income €{user['user_income']:,}, Age {user['user_age']}")
        print(f"   Predicted Profile: {profile}")
        print(f"   Probabilities: {', '.join([f'{k}: {v:.2f}' for k, v in probs.items()])}")
        print(f"   Search Weights: Cost={weights['cost']:.2f}, Time={weights['time']:.2f}, Comfort={weights['comfort']:.2f}")

        # Explanation
        explanation = classifier.explain_prediction(profile, probs)
        print(f"   Explanation: {explanation.split('Caratteristiche')[0].strip()}")

    print(f"\n[DONE] User Preference Classifier trainato e testato!")