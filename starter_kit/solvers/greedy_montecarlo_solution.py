import json
import math
import random
import copy
from score_function import getSolutionScore


# Définition des types d'antennes
ANTENNA_TYPES = {
    "Nano": {
        "range": 50,
        "capacity": 200,
        "cost_on_building": 5_000,
        "cost_off_building": 6_000,
    },
    "Spot": {
        "range": 100,
        "capacity": 800,
        "cost_on_building": 15_000,
        "cost_off_building": 20_000,
    },
    "Density": {
        "range": 150,
        "capacity": 5_000,
        "cost_on_building": 30_000,
        "cost_off_building": 50_000,
    },
    "MaxRange": {
        "range": 400,
        "capacity": 3_500,
        "cost_on_building": 40_000,
        "cost_off_building": 50_000,
    },
}


def calculate_distance(x1, y1, x2, y2):
    """Calcule la distance euclidienne entre deux points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_max_population(building):
    """Retourne la population maximale d'un bâtiment sur les 3 périodes."""
    return max(
        building["populationPeakHours"],
        building["populationOffPeakHours"],
        building["populationNight"],
    )


def get_buildings_in_range(
    antenna_x, antenna_y, antenna_type, buildings, available_buildings
):
    """Retourne la liste des bâtiments disponibles dans la portée de l'antenne."""
    antenna_range = ANTENNA_TYPES[antenna_type]["range"]
    in_range = []

    for building_id in available_buildings:
        building = buildings[building_id]
        distance = calculate_distance(
            antenna_x, antenna_y, building["x"], building["y"]
        )
        if distance <= antenna_range:
            in_range.append(building_id)

    return in_range


def select_buildings_within_capacity(x, y, antenna_type, buildings, in_range):
    """
    Sélectionne les bâtiments dans la portée en respectant la capacité de l'antenne.
    Retourne la liste des bâtiments sélectionnés.
    """
    specs = ANTENNA_TYPES[antenna_type]

    # Vérifier la capacité
    total_pop = sum(get_max_population(buildings[bid]) for bid in in_range)
    if total_pop <= specs["capacity"]:
        return in_range

    # Réduire les bâtiments pour respecter la capacité
    # Tri par distance croissante
    sorted_buildings = sorted(
        in_range,
        key=lambda bid: calculate_distance(
            x, y, buildings[bid]["x"], buildings[bid]["y"]
        ),
    )
    selected = []
    current_pop = 0
    for bid in sorted_buildings:
        building_pop = get_max_population(buildings[bid])
        if current_pop + building_pop <= specs["capacity"]:
            selected.append(bid)
            current_pop += building_pop

    return selected


def greedy_solution(buildings_dict, available_buildings, efficiency_func=None):
    """
    Algorithme greedy pour placer les antennes.

    Args:
        buildings_dict: Dictionnaire des bâtiments
        available_buildings: Set des bâtiments à couvrir
        efficiency_func: Fonction pour calculer l'efficacité (optionnel)

    Returns:
        Liste d'antennes
    """
    available = available_buildings.copy()
    antennas = []

    # Fonction d'efficacité par défaut
    if efficiency_func is None:
        efficiency_func = lambda n_buildings, cost: (n_buildings**1.5) / cost

    while available:
        best_choice = None
        best_efficiency = 0

        # Positions candidates : sur les bâtiments + barycentre
        candidate_positions = []

        # Positions sur chaque bâtiment disponible
        for building_id in available:
            building = buildings_dict[building_id]
            candidate_positions.append((building["x"], building["y"], True))

        # Position centrale (barycentre)
        if available:
            avg_x = sum(buildings_dict[bid]["x"] for bid in available) / len(available)
            avg_y = sum(buildings_dict[bid]["y"] for bid in available) / len(available)
            candidate_positions.append((int(avg_x), int(avg_y), False))

        # Tester chaque position candidate avec chaque type d'antenne
        for x, y, is_on_building in candidate_positions:
            for antenna_type in ANTENNA_TYPES.keys():
                # Trouver les bâtiments dans la portée
                in_range = get_buildings_in_range(
                    x, y, antenna_type, buildings_dict, available
                )

                if not in_range:
                    continue

                # Sélectionner les bâtiments dans la capacité
                selected = select_buildings_within_capacity(
                    x, y, antenna_type, buildings_dict, in_range
                )

                if not selected:
                    continue

                # Calculer le coût
                specs = ANTENNA_TYPES[antenna_type]
                cost = (
                    specs["cost_on_building"]
                    if is_on_building
                    else specs["cost_off_building"]
                )

                # Calculer l'efficacité
                efficiency = efficiency_func(len(selected), cost)

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_choice = {
                        "type": antenna_type,
                        "x": x,
                        "y": y,
                        "buildings": selected,
                        "cost": cost,
                        "efficiency": efficiency,
                        "is_on_building": is_on_building,
                    }

        if not best_choice:
            print("⚠️  Impossible de couvrir les bâtiments restants!")
            break

        # Placer la meilleure antenne
        antennas.append(
            {
                "type": best_choice["type"],
                "x": best_choice["x"],
                "y": best_choice["y"],
                "buildings": best_choice["buildings"],
            }
        )

        # Retirer les bâtiments couverts
        available -= set(best_choice["buildings"])

    return antennas


def monte_carlo_optimization(dataset, n_iterations=100, temperature=0.3):
    """
    Optimisation Monte Carlo avec multiples stratégies greedy.

    Idée :
    - Essayer différentes fonctions d'efficacité pour le greedy
    - Essayer des variations randomisées de l'ordre de traitement
    - Garder la meilleure solution trouvée

    Args:
        dataset: Dataset des bâtiments
        n_iterations: Nombre d'itérations Monte Carlo
        temperature: Paramètre de randomisation (0-1)
    """
    buildings = {b["id"]: b for b in dataset["buildings"]}
    available_buildings = set(buildings.keys())

    best_solution = None
    best_cost = float("inf")

    print(f"\n{'='*70}")
    print(f"MONTE CARLO + GREEDY")
    print(f"{'='*70}")
    print(f"Bâtiments à couvrir : {len(available_buildings)}")
    print(f"Nombre d'itérations : {n_iterations}")
    print(f"Température : {temperature}")

    # Définir différentes fonctions d'efficacité
    efficiency_functions = [
        # Standard : buildings^1.5 / cost
        lambda n, c: (n**1.5) / c,
        # Favorise beaucoup les grandes couvertures
        lambda n, c: (n**2) / c,
        # Favorise un peu moins les grandes couvertures
        lambda n, c: (n**1.3) / c,
        # Simple ratio
        lambda n, c: n / c,
        # Favorise le nombre de bâtiments avec pénalité légère sur le coût
        lambda n, c: (n**1.8) / (c**0.9),
        # Maximise le nombre de bâtiments
        lambda n, c: n,
    ]

    for iteration in range(n_iterations):
        # Choisir une fonction d'efficacité aléatoirement
        efficiency_func = random.choice(efficiency_functions)

        # Ajouter une composante aléatoire à l'efficacité
        if temperature > 0:
            original_func = efficiency_func
            efficiency_func = lambda n, c, of=original_func: of(n, c) * (
                1 + random.uniform(-temperature, temperature)
            )

        # Générer une solution greedy avec cette fonction
        antennas = greedy_solution(buildings, available_buildings, efficiency_func)

        # Calculer le coût
        solution = {"antennas": antennas}
        cost, is_valid, message = getSolutionScore(
            json.dumps(solution), json.dumps(dataset)
        )

        if is_valid and cost < best_cost:
            best_cost = cost
            best_solution = solution
            print(
                f"🎯 Itération {iteration + 1}/{n_iterations}: Nouvelle meilleure solution ! Coût = {cost:,} € ({len(antennas)} antennes)"
            )
        elif iteration % 10 == 0:
            status = "✓" if is_valid else "✗"
            print(
                f"   Itération {iteration + 1}/{n_iterations}: {status} Coût = {cost:,} € ({len(antennas)} antennes)"
            )

    return best_solution


def local_search_optimization(initial_solution, dataset, n_iterations=50):
    """
    Recherche locale pour améliorer une solution existante.

    Stratégies :
    - Essayer de remplacer une antenne par une autre (même position, différent type)
    - Essayer de déplacer une antenne
    - Essayer de fusionner deux antennes en une seule
    """
    buildings = {b["id"]: b for b in dataset["buildings"]}
    current_solution = copy.deepcopy(initial_solution)

    cost, is_valid, _ = getSolutionScore(
        json.dumps(current_solution), json.dumps(dataset)
    )
    if not is_valid:
        return initial_solution

    best_cost = cost
    best_solution = current_solution

    print(f"\n{'='*70}")
    print(f"RECHERCHE LOCALE")
    print(f"{'='*70}")
    print(
        f"Solution initiale : {best_cost:,} € ({len(current_solution['antennas'])} antennes)"
    )

    for iteration in range(n_iterations):
        improved = False

        # Essayer de modifier chaque antenne
        for i, antenna in enumerate(current_solution["antennas"]):
            # Stratégie 1 : Changer le type d'antenne
            for new_type in ANTENNA_TYPES.keys():
                if new_type == antenna["type"]:
                    continue

                # Créer une solution modifiée
                test_solution = copy.deepcopy(current_solution)
                test_solution["antennas"][i]["type"] = new_type

                # Vérifier si les bâtiments sont toujours couverts
                x, y = antenna["x"], antenna["y"]
                in_range = get_buildings_in_range(
                    x, y, new_type, buildings, set(antenna["buildings"])
                )
                selected = select_buildings_within_capacity(
                    x, y, new_type, buildings, in_range
                )

                if set(selected) == set(antenna["buildings"]):
                    # Les mêmes bâtiments sont couverts, tester le coût
                    test_cost, test_valid, _ = getSolutionScore(
                        json.dumps(test_solution), json.dumps(dataset)
                    )

                    if test_valid and test_cost < best_cost:
                        best_cost = test_cost
                        best_solution = test_solution
                        current_solution = test_solution
                        improved = True
                        print(
                            f"  ✓ Amélioration (changement type antenne {i}): {test_cost:,} €"
                        )

        if not improved:
            break

    print(f"\nSolution finale après recherche locale : {best_cost:,} €")
    return best_solution


def main():
    import sys

    # Récupérer le numéro de dataset depuis les arguments
    if len(sys.argv) > 1:
        dataset_num = sys.argv[1]
    else:
        dataset_num = "3"

    # Mapping des noms de datasets
    dataset_names = {
        "1": "1_peaceful_village",
        "2": "2_small_town",
        "3": "3_suburbia",
        "4": "4_epitech",
        "5": "5_isogrid",
        "6": "6_manhattan",
    }

    dataset_name = dataset_names.get(dataset_num, "3_suburbia")
    input_file = f"./datasets/{dataset_name}.json"

    print(f"\n{'='*70}")
    print(f"SOLUTION GREEDY + MONTE CARLO")
    print(f"Dataset : {dataset_name}.json")
    print(f"{'='*70}")

    print(f"\nChargement du dataset : {input_file}")
    dataset = json.load(open(input_file))

    print(f"\nAnalyse du dataset :")
    print(f"Nombre de bâtiments : {len(dataset['buildings'])}")
    total_pop = sum(get_max_population(b) for b in dataset["buildings"])
    print(f"Population totale : {total_pop}")

    # Phase 1 : Monte Carlo avec greedy (plus d'itérations pour grand dataset)
    solution_mc = monte_carlo_optimization(dataset, n_iterations=200, temperature=0.3)

    if solution_mc:
        cost_mc, is_valid_mc, message_mc = getSolutionScore(
            json.dumps(solution_mc), json.dumps(dataset)
        )
        print(f"\n{message_mc}")

        # Phase 2 : Recherche locale (désactivée pour grand dataset)
        print(f"\nRecherche locale ignorée pour les grands datasets (>100 bâtiments)")
        solution_final = solution_mc

        cost_final, is_valid_final, message_final = getSolutionScore(
            json.dumps(solution_final), json.dumps(dataset)
        )

        print(f"\n{'='*70}")
        print(f"RÉSULTAT FINAL")
        print(f"{'='*70}")
        print(f"{message_final}")

        if is_valid_final:
            output_file = f"./solutions/{dataset_name}/solution_{dataset_name}_{cost_final}_greedy_mc.json"
            with open(output_file, "w") as f:
                json.dump(solution_final, f, indent=2)
            print(f"\n✓ Solution sauvegardée dans {output_file}")

            # Afficher les détails de la solution
            print(f"\nDétails de la solution :")
            antenna_counts = {}
            for antenna in solution_final["antennas"]:
                antenna_type = antenna["type"]
                antenna_counts[antenna_type] = antenna_counts.get(antenna_type, 0) + 1

            for antenna_type, count in sorted(antenna_counts.items()):
                print(f"  - {antenna_type}: {count} antenne(s)")

            # Comparaison avec la solution naïve
            naive_cost = len(dataset["buildings"]) * 30_000  # Une Density par bâtiment
            improvement = ((naive_cost - cost_final) / naive_cost) * 100
            print(
                f"\nAmélioration vs solution naïve : {improvement:.1f}% ({naive_cost:,} € → {cost_final:,} €)"
            )
        else:
            print(f"\n✗ Solution invalide : {message_final}")
    else:
        print("\n✗ Aucune solution valide trouvée")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
