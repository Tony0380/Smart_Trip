# SMART-TRIP: Sistema Multi-paradigma per Pianificazione Viaggi Intelligente

Sistema di pianificazione viaggi che integra quattro paradigmi dell'Intelligenza Artificiale: algoritmi di ricerca, machine learning, ragionamento probabilistico e rappresentazione della conoscenza logica.

**Autore**: Antonio Colamartino (a.colamartino6@studenti.uniba.it)  
**Corso**: Ingegneria della Conoscenza 2024-2025  
**Università**: Università di Bari "Aldo Moro"

## Struttura del Progetto

```
├── intelligent_travel_planner.py      # Orchestratore principale del sistema
├── test_complete_system.py           # Test di integrazione completo
├── data_collection/
│   └── transport_data.py             # Grafo delle città italiane (NetworkX)
├── search_algorithms/
│   └── pathfinder.py                 # A*, Floyd-Warshall, Dijkstra, Beam Search
├── ml_models/
│   ├── predictor_models.py           # Predizione prezzi viaggi
│   ├── preference_classifier.py      # Classificazione profili utente
│   ├── dataset_generator.py          # Generazione dataset sintetici
│   └── ml_pathfinder_integration.py  # Integrazione ML con algoritmi di ricerca
├── bayesian_network/
│   └── uncertainty_models.py         # Rete Bayesiana per gestione incertezza
├── prolog_kb/
│   ├── travel_rules.pl               # Knowledge Base Prolog
│   └── prolog_interface.py           # Interfaccia Python-SWI-Prolog
└── evaluation/
    └── comprehensive_evaluation.py   # Framework di valutazione performance
```

## Requisiti di Sistema

### Dipendenze Python
```bash
pip install -r requirements.txt
```

Dipendenze principali:
- numpy >= 1.24.0 (calcoli numerici)
- pandas >= 2.0.0 (manipolazione dati)
- scikit-learn >= 1.3.0 (algoritmi ML)
- networkx >= 3.0.0 (algoritmi grafi)
- joblib >= 1.3.0 (serializzazione modelli)

### SWI-Prolog (Opzionale)
Il sistema include un fallback engine Python per la KB Prolog, ma per performance ottimali:
- **Windows**: Scaricare da https://www.swi-prolog.org/download/stable
- **Linux**: `sudo apt-get install swi-prolog`
- **macOS**: `brew install swi-prolog`

Il sistema funziona anche senza SWI-Prolog usando il motore di inferenza Python integrato.

## Esecuzione

### Demo Interattivo
```bash
python intelligent_travel_planner.py --demo
```
Esegue 3 scenari predefiniti (business, budget, leisure) mostrando il funzionamento completo del sistema.

### Pianificazione Personalizzata
```bash
# Pianificazione base
python intelligent_travel_planner.py --plan milano roma --budget 200

# Con profilo utente specifico
python intelligent_travel_planner.py --plan venezia napoli --profile business --budget 300 --season winter

# Con condizioni meteo
python intelligent_travel_planner.py --plan torino palermo --profile budget --budget 150 --weather Bad
```

Parametri disponibili:
- `--plan ORIGINE DESTINAZIONE`: Città di partenza e arrivo
- `--budget IMPORTO`: Budget massimo in euro (default: 500)
- `--profile TIPO`: business, leisure, o budget
- `--season STAGIONE`: summer, winter, spring, autumn (default: summer)
- `--weather CONDIZIONI`: Good, Fair, Bad (default: Fair)

### Valutazione Performance
```bash
python intelligent_travel_planner.py --evaluate
```
Esegue benchmark completo con K-fold cross-validation e salva risultati dettagliati.

### Performance Ottenute
- **Price Predictor**: R² = 0.823 (Gradient Boosting)
- **User Classifier**: Accuracy = 100% (Logistic Regression)  
- **Time Estimator**: R² = 0.888 ± 0.115
- **System Response**: < 1 secondo per pianificazione singola

### Test Sistema Completo
```bash
python test_complete_system.py
```
Verifica il corretto funzionamento di tutti i moduli integrati.

### Modalità Silenziosa
```bash
python intelligent_travel_planner.py --plan milano roma --quiet
```
Riduce l'output per utilizzi automatizzati.

### Senza Training
```bash
python intelligent_travel_planner.py --plan milano roma --no-training
```
Usa modelli pre-trainati (se disponibili) invece di ri-addestrare.

## Città Supportate

Il sistema supporta 20 città italiane principali:
Milano, Roma, Napoli, Torino, Bologna, Firenze, Bari, Palermo, Venezia, Genova, Verona, Catania, Cagliari, Trieste, Perugia, Pescara, Reggio Calabria, Salerno, Brescia, Pisa.

## Paradigmi AI Implementati

1. **Search Algorithms**: A* multi-obiettivo, Floyd-Warshall, Dijkstra per ottimizzazione percorsi
2. **Machine Learning**: Gradient Boosting per predizione prezzi, Random Forest per classificazione profili utente
3. **Probabilistic Reasoning**: Rete Bayesiana con 6 nodi per gestione incertezza meteo e trasporti
4. **Logic Programming**: Knowledge Base Prolog avanzata con inferenza complessa:
   - Multi-hop pathfinding con ricerca dinamica fino a 5 salti
   - Calcolo affidabilità considerando eventi e condizioni meteorologiche
   - Meta-ragionamento con spiegazioni automatiche delle decisioni
   - Constraint satisfaction con propagazione vincoli
   - Ottimizzazione multi-obiettivo con ranking dinamico
   - Pianificazione temporale con disponibilità oraria trasporti

## Output del Sistema

Il sistema fornisce:
- **Percorso raccomandato** con costo, tempo e distanza totali (es. Milano → Bologna → Firenze → Roma, 79.42€, 6.5h)
- **Analisi del profilo utente** con classificazione automatica (business/leisure/budget) e confidence score
- **Validazione vincoli** tramite KB Prolog avanzata con inferenza multi-livello
- **Analisi incertezza** Bayesiana (Train: Success=89%, Bus: Success=90%, Flight: Success=83%)
- **Spiegazioni interpretabili** del processo decisionale con meta-ragionamento
- **Performance metriche**: Price Predictor R²=0.823, User Classifier Accuracy=100%, Time Estimator R²=0.888
