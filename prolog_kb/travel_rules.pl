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

% Caratteristiche trasporti
transport(train, 120, 0.15, high, medium).
transport(bus, 80, 0.08, low, low).
transport(flight, 500, 0.25, high, high).

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

% REGOLE PRINCIPALI

% Collegamento bidirezionale
connected(A, B, Transports, Distance) :-
    connection(A, B, Transports, Distance).
connected(A, B, Transports, Distance) :-
    connection(B, A, Transports, Distance).

% Percorso diretto
path(Origin, Dest, [Origin, Dest], Transport, Cost) :-
    connected(Origin, Dest, Transports, Distance),
    member(Transport, Transports),
    transport(Transport, _, CostKm, _, _),
    Cost is Distance * CostKm.

% Percorso con un intermedio
path(Origin, Dest, [Origin, Mid, Dest], Transport, TotalCost) :-
    connected(Origin, Mid, T1, D1),
    connected(Mid, Dest, T2, D2),
    member(Transport, T1),
    member(Transport, T2),
    Origin \= Dest,
    Mid \= Origin,
    Mid \= Dest,
    transport(Transport, _, CostKm, _, _),
    TotalCost is (D1 + D2) * CostKm.

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

% Ordinamenti semplificati
sort_by_speed([flight, train, bus], [flight, train, bus]).
sort_by_speed([train, bus], [train, bus]).
sort_by_speed([flight], [flight]).
sort_by_speed([bus], [bus]).

sort_by_cost([bus, train, flight], [bus, train, flight]).
sort_by_cost([train, flight], [train, flight]).
sort_by_cost([bus], [bus]).

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