% Travel Planning Knowledge Base - ICon 2024/25 (Human-Optimized)
% 
% Rappresentazione conoscenza dominio viaggio essenziale per
% pianificazione intelligente e constraint satisfaction
% 
% ARGOMENTI DEL PROGRAMMA IMPLEMENTATI:
% 1. Logica del primo ordine e clausole di Horn
% 2. Unificazione e risoluzione SLD  
% 3. Backward chaining per goal resolution
% 4. Cut e negation as failure

% FATTI BASE: Città italiane principali

% Città con caratteristiche essenziali
city(milano, north, big, high_cost).
city(roma, center, big, medium_cost).  
city(napoli, south, big, medium_cost).
city(torino, north, medium, medium_cost).
city(bologna, north, medium, medium_cost).
city(firenze, center, medium, medium_cost).
city(bari, south, medium, low_cost).
city(palermo, south, big, low_cost).
city(venezia, north, small, high_cost).
city(genova, north, medium, medium_cost).
city(verona, north, small, medium_cost).
city(catania, south, medium, low_cost).
city(cagliari, south, small, low_cost).
city(perugia, center, small, medium_cost).
city(pisa, center, small, medium_cost).

% Collegamenti diretti principali (Nord)
connection(milano, torino, [train, bus], 140).
connection(milano, venezia, [train, bus], 280).
connection(milano, bologna, [train, bus], 200).
connection(milano, genova, [train, bus], 120).
connection(torino, genova, [train, bus], 95).
connection(venezia, verona, [train, bus], 80).
connection(verona, bologna, [train, bus], 150).

% Collegamenti Centro
connection(bologna, firenze, [train, bus], 105).
connection(firenze, roma, [train, bus], 270).
connection(firenze, pisa, [train, bus], 85).
connection(perugia, roma, [train, bus], 180).

% Collegamenti Sud  
connection(roma, napoli, [train, bus], 225).
connection(napoli, bari, [train, bus], 280).

% Collegamenti aerei lunghe distanze
connection(milano, palermo, [flight], 935).
connection(milano, cagliari, [flight], 850).
connection(roma, palermo, [flight], 490).
connection(roma, cagliari, [flight], 470).
connection(milano, bari, [flight], 770).
connection(roma, bari, [flight], 450).

% Caratteristiche trasporti con disponibilità temporale
transport(train, 120, 0.15, high, medium, [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]).
transport(bus, 80, 0.08, low, low, [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]).
transport(flight, 500, 0.25, high, high, [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]).

% Condizioni meteorologiche per città
weather_impact(north, winter, [train:0.8, bus:0.6, flight:0.9]).
weather_impact(north, summer, [train:1.0, bus:1.0, flight:1.0]).
weather_impact(south, winter, [train:0.9, bus:0.8, flight:0.95]).
weather_impact(south, summer, [train:0.95, bus:0.9, flight:1.0]).
weather_impact(center, _, [train:0.95, bus:0.9, flight:0.98]).

% Capacità dei trasporti
capacity(train, 300, high_frequency).
capacity(bus, 50, medium_frequency).
capacity(flight, 180, low_frequency).

% Eventi che influenzano disponibilità
event(strike, train, [probability:0.05, impact:0.0]).
event(delay, flight, [probability:0.15, impact:0.8]).
event(roadwork, bus, [probability:0.1, impact:0.7]).

% PROFILI UTENTE

% Profili con caratteristiche comportamentali
profile(business, high_income, low_price_sens, high_time, high_comfort).
profile(leisure, medium_income, medium_price_sens, medium_time, medium_comfort).
profile(budget, low_income, high_price_sens, low_time, low_comfort).
profile(family, medium_income, medium_price_sens, high_time, high_comfort).
profile(student, low_income, very_high_price_sens, low_time, low_comfort).

% Preferenze trasporto per profilo
prefers(business, flight).
prefers(business, train).
prefers(leisure, train).
prefers(budget, bus).
prefers(family, train).
prefers(student, bus).

% REGOLE PRINCIPALI CON INFERENZA COMPLESSA

% Collegamento bidirezionale
connected(A, B, Transports, Distance) :-
    connection(A, B, Transports, Distance).
connected(A, B, Transports, Distance) :-
    connection(B, A, Transports, Distance).

% Percorso multi-hop con ricerca dinamica
path(Origin, Dest, Path, Transport, TotalCost) :-
    path(Origin, Dest, [Origin], Path, Transport, TotalCost, 0, 5).

path(Dest, Dest, Acc, Path, _, 0, _, _) :-
    reverse(Acc, Path).

path(Current, Dest, Acc, Path, Transport, TotalCost, CostSoFar, MaxHops) :-
    MaxHops > 0,
    connected(Current, Next, Transports, Distance),
    member(Transport, Transports),
    \+ member(Next, Acc),
    transport(Transport, _, CostKm, _, _, _),
    SegmentCost is Distance * CostKm,
    NewCost is CostSoFar + SegmentCost,
    NewHops is MaxHops - 1,
    path(Next, Dest, [Next|Acc], Path, Transport, TotalCost, NewCost, NewHops),
    TotalCost = NewCost.

% Inferenza su disponibilità dinamica
available_at_time(Transport, Hour) :-
    transport(Transport, _, _, _, _, AvailableHours),
    member(Hour, AvailableHours).

% Calcolo affidabilità considerando eventi
transport_reliability(Transport, Region, Season, Reliability) :-
    findall(Impact, (event(_, Transport, Props), member(impact:Impact, Props)), Impacts),
    findall(Prob, (event(_, Transport, Props), member(probability:Prob, Props)), Probs),
    calculate_reliability(Impacts, Probs, BaseReliability),
    weather_impact(Region, Season, WeatherEffects),
    member(Transport:WeatherFactor, WeatherEffects),
    Reliability is BaseReliability * WeatherFactor.

calculate_reliability([], [], 1.0).
calculate_reliability([Impact|Impacts], [Prob|Probs], Reliability) :-
    calculate_reliability(Impacts, Probs, SubReliability),
    Reliability is SubReliability * (1 - Prob * (1 - Impact)).

% Ottimizzazione multi-obiettivo con inferenza
optimal_route(Origin, Dest, Profile, Criteria, OptimalRoute) :-
    findall(route(Path, Transport, Cost, Time, Comfort, Reliability),
            (path(Origin, Dest, Path, Transport, Cost),
             calculate_route_metrics(Path, Transport, Profile, Time, Comfort, Reliability)),
            Routes),
    rank_routes(Routes, Criteria, [OptimalRoute|_]).

calculate_route_metrics(Path, Transport, Profile, Time, Comfort, Reliability) :-
    path_length(Path, Length),
    transport(Transport, Speed, _, ComfortLevel, _, _),
    Time is Length / Speed * 60,
    map_comfort(ComfortLevel, Comfort),
    city(Origin, Region, _, _),
    current_season(Season),
    transport_reliability(Transport, Region, Season, Reliability).

map_comfort(high, 0.9).
map_comfort(medium, 0.6).
map_comfort(low, 0.3).

path_length([_], 0).
path_length([A,B|Rest], TotalLength) :-
    connected(A, B, _, Distance),
    path_length([B|Rest], RestLength),
    TotalLength is Distance + RestLength.

% Constraint satisfaction con propagazione
satisfies_all_constraints(Route, Constraints) :-
    forall(member(Constraint, Constraints),
           satisfies_constraint(Route, Constraint)).

satisfies_constraint(route(_, _, Cost, _, _, _), budget(Budget)) :-
    Cost =< Budget.
satisfies_constraint(route(_, _, _, Time, _, _), max_time(MaxTime)) :-
    Time =< MaxTime.
satisfies_constraint(route(_, _, _, _, Comfort, _), min_comfort(MinComfort)) :-
    Comfort >= MinComfort.
satisfies_constraint(route(_, _, _, _, _, Reliability), min_reliability(MinReliability)) :-
    Reliability >= MinReliability.

% Meta-ragionamento: spiegazione delle decisioni
explain_route_choice(route(Path, Transport, Cost, Time, Comfort, Reliability), Profile, Explanation) :-
    findall(Reason, generate_reason(Path, Transport, Cost, Time, Comfort, Reliability, Profile, Reason), Explanation).

generate_reason(_, Transport, _, _, _, _, business, 'Selected for speed priority') :-
    transport(Transport, Speed, _, _, _, _), Speed > 400.
generate_reason(_, Transport, Cost, _, _, _, budget, 'Selected for cost efficiency') :-
    transport(Transport, _, CostKm, _, _, _), CostKm < 0.1.
generate_reason(Path, _, _, _, _, Reliability, _, 'Reliable route chosen') :-
    Reliability > 0.8, length(Path, Len), Len =< 3.

% Pianificazione temporale con vincoli
schedule_trip(Origin, Dest, Profile, DepartureTime, ArrivalTime, Schedule) :-
    path(Origin, Dest, Path, Transport, _),
    available_at_time(Transport, DepartureTime),
    calculate_travel_time(Path, Transport, TravelTime),
    ArrivalTime is DepartureTime + TravelTime,
    Schedule = schedule(departure:DepartureTime, arrival:ArrivalTime, transport:Transport, path:Path).

calculate_travel_time(Path, Transport, TotalTime) :-
    path_length(Path, Distance),
    transport(Transport, Speed, _, _, _, _),
    TotalTime is Distance / Speed * 60.

current_season(summer).

% Trasporto disponibile
available(Origin, Dest, Transport) :-
    connected(Origin, Dest, Transports, _),
    member(Transport, Transports).

% Trasporto adatto per profilo
suitable(Profile, Transport) :-
    profile(Profile, Income, PriceSens, TimePriority, ComfortPriority),
    transport(Transport, Speed, Cost, Comfort, Reliability),
    match_profile(Income, PriceSens, TimePriority, ComfortPriority, 
                  Transport, Speed, Cost, Comfort, Reliability).

% Matching regole per profili
match_profile(high_income, _, high_time, high_comfort, flight, Speed, _, high, _) :- 
    Speed >= 400.
match_profile(high_income, _, high_time, _, train, Speed, _, _, high) :- 
    Speed >= 100.
match_profile(_, high_price_sens, _, _, bus, _, Cost, _, _) :- 
    Cost =< 0.10.
match_profile(medium_income, medium_price_sens, _, _, train, Speed, Cost, _, _) :- 
    Speed >= 80, Cost =< 0.20.

% Viaggio valido
valid_trip(Origin, Dest, Profile, Transport, Budget) :-
    path(Origin, Dest, _, Transport, Cost),
    suitable(Profile, Transport),
    Cost =< Budget,
    no_restrictions(Origin, Dest).

% Budget check
within_budget(Origin, Dest, Transport, Budget) :-
    path(Origin, Dest, _, Transport, Cost),
    Cost =< Budget.

% Restrizioni di viaggio
no_restrictions(_, _).  % Semplificato per versione ottimizzata

% Miglior trasporto per utente
best_transport(Origin, Dest, Profile, BestTransport) :-
    findall(Transport, 
            (available(Origin, Dest, Transport), suitable(Profile, Transport)), 
            Suitable),
    Suitable \= [],
    rank_by_profile(Suitable, Profile, [BestTransport|_]).

% Ranking trasporti
rank_by_profile(Transports, business, Ranked) :-
    sort_by_speed(Transports, Ranked).
rank_by_profile(Transports, budget, Ranked) :-
    sort_by_cost(Transports, Ranked).
rank_by_profile(Transports, _, [train, bus, flight]).

% Ranking dinamico basato su metriche calcolate
rank_routes(Routes, Criteria, RankedRoutes) :-
    maplist(score_route(Criteria), Routes, ScoredRoutes),
    keysort(ScoredRoutes, Sorted),
    reverse(Sorted, ReverseSorted),
    pairs_values(ReverseSorted, RankedRoutes).

score_route(Criteria, Route, Score-Route) :-
    Route = route(_, _, Cost, Time, Comfort, Reliability),
    normalize_metrics(Cost, Time, Comfort, Reliability, NCost, NTime, NComfort, NReliability),
    calculate_weighted_score(Criteria, NCost, NTime, NComfort, NReliability, Score).

normalize_metrics(Cost, Time, Comfort, Reliability, NCost, NTime, NComfort, NReliability) :-
    NCost is 1 / (1 + Cost / 100),
    NTime is 1 / (1 + Time / 60),
    NComfort is Comfort,
    NReliability is Reliability.

calculate_weighted_score(business, NCost, NTime, NComfort, NReliability, Score) :-
    Score is 0.1 * NCost + 0.4 * NTime + 0.25 * NComfort + 0.25 * NReliability.
calculate_weighted_score(budget, NCost, NTime, NComfort, NReliability, Score) :-
    Score is 0.5 * NCost + 0.2 * NTime + 0.1 * NComfort + 0.2 * NReliability.
calculate_weighted_score(leisure, NCost, NTime, NComfort, NReliability, Score) :-
    Score is 0.25 * NCost + 0.2 * NTime + 0.35 * NComfort + 0.2 * NReliability.

% Multi-città trip
multi_trip([Start|Cities], Profile, Budget, Plan) :-
    plan_segments(Start, Cities, Profile, Budget, [], Plan).

plan_segments(_, [], _, _, Acc, Acc).
plan_segments(Current, [Next|Rest], Profile, Budget, Acc, Plan) :-
    path(Current, Next, Path, Transport, Cost),
    suitable(Profile, Transport),
    Budget >= Cost,
    NewBudget is Budget - Cost,
    append(Acc, [(Current, Next, Transport, Cost)], NewAcc),
    plan_segments(Next, Rest, Profile, NewBudget, NewAcc, Plan).

% Tutti i percorsi possibili con costi
all_routes(Origin, Dest, Profile, Budget, Routes) :-
    findall((Path, Transport, Cost), 
            (path(Origin, Dest, Path, Transport, Cost),
             Cost =< Budget,
             suitable(Profile, Transport)),
            Routes).

% Pianificazione viaggio completa
plan_travel(Origin, Dest, Profile, Budget, Plan) :-
    all_routes(Origin, Dest, Profile, Budget, Routes),
    Routes \= [],
    select_best_route(Routes, Profile, BestRoute),
    BestRoute = (Path, Transport, Cost),
    Plan = travel_plan(route:Path, transport:Transport, cost:Cost).

% Selezione miglior percorso
select_best_route([(Path, Transport, Cost)], _, (Path, Transport, Cost)).
select_best_route([(P1,T1,C1), (P2,T2,C2)|Rest], Profile, Best) :-
    (   prefer_route((P1,T1,C1), (P2,T2,C2), Profile) ->
        select_best_route([(P1,T1,C1)|Rest], Profile, Best)
    ;   select_best_route([(P2,T2,C2)|Rest], Profile, Best)
    ).

% Preferenza tra percorsi
prefer_route((_, T1, C1), (_, T2, C2), business) :-
    transport(T1, S1, _, _, _),
    transport(T2, S2, _, _, _),
    S1 > S2.
prefer_route((_, _, C1), (_, _, C2), budget) :-
    C1 =< C2.
prefer_route((_, T1, C1), (_, T2, C2), _) :-
    C1 =< C2.

% Consigli viaggio
advice(Origin, Dest, Profile, Advice) :-
    findall(T, available(Origin, Dest, T), Available),
    findall(T, (member(T, Available), suitable(Profile, T)), Suitable),
    (   Suitable \= [] ->
        Advice = recommended(Suitable)
    ;   Advice = alternatives(Available)
    ).

% Fattibilità viaggio
feasible(Origin, Dest, Profile, Budget, Result) :-
    (   valid_trip(Origin, Dest, Profile, _, Budget) ->
        Result = feasible
    ;   findall(C, path(Origin, Dest, _, _, C), Costs),
        min_list(Costs, MinCost),
        (   MinCost > Budget ->
            Result = insufficient_budget(MinCost)
        ;   Result = no_suitable_transport
        )
    ).

% Query per testing
test_basic :-
    connected(milano, roma, _, _),
    write('Milano-Roma: connected'), nl.

test_planning :-
    plan_travel(milano, napoli, business, 200, Plan),
    write('Business trip Milano-Napoli: '), write(Plan), nl.

% Fine Knowledge Base