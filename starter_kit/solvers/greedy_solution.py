import json
import math
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


def get_best_antenna_for_position(x, y, buildings, available_buildings, all_buildings):
    """
    Trouve le meilleur type d'antenne pour une position donnée.
    Retourne (type, buildings_covered, cost_efficiency)
    """
    is_on_building = any(
        all_buildings[bid]["x"] == x and all_buildings[bid]["y"] == y
        for bid in range(len(all_buildings))
    )

    best_option = None
    best_efficiency = 0  # buildings covered per euro

    for antenna_type, specs in ANTENNA_TYPES.items():
        # Trouver les bâtiments dans la portée
        in_range = get_buildings_in_range(
            x, y, antenna_type, all_buildings, available_buildings
        )

        if not in_range:
            continue

        # Vérifier la capacité
        total_pop = sum(get_max_population(all_buildings[bid]) for bid in in_range)
        if total_pop > specs["capacity"]:
            # Réduire les bâtiments pour respecter la capacité
            # Tri par distance croissante
            sorted_buildings = sorted(
                in_range,
                key=lambda bid: calculate_distance(
                    x, y, all_buildings[bid]["x"], all_buildings[bid]["y"]
                ),
            )
            in_range = []
            current_pop = 0
            for bid in sorted_buildings:
                building_pop = get_max_population(all_buildings[bid])
                if current_pop + building_pop <= specs["capacity"]:
                    in_range.append(bid)
                    current_pop += building_pop

        if not in_range:
            continue

        # Calculer le coût
        cost = (
            specs["cost_on_building"] if is_on_building else specs["cost_off_building"]
        )

        # Efficacité : nombre de bâtiments couverts par euro
        efficiency = len(in_range) / cost

        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_option = (antenna_type, in_range, cost)

    return best_option


def get_candidate_positions(buildings, available_buildings):
    """
    Génère des positions candidates pour placer une antenne.
    Retourne les positions des bâtiments + quelques positions centrales calculées.
    """
    positions = []

    # Positions sur chaque bâtiment disponible
    for building_id in available_buildings:
        building = buildings[building_id]
        positions.append((building["x"], building["y"], True))  # True = sur bâtiment

    # Position centrale (barycentre)
    if available_buildings:
        avg_x = sum(buildings[bid]["x"] for bid in available_buildings) / len(
            available_buildings
        )
        avg_y = sum(buildings[bid]["y"] for bid in available_buildings) / len(
            available_buildings
        )
        positions.append((int(avg_x), int(avg_y), False))  # False = hors bâtiment

    return positions


def greedy_solution(dataset):
    """
    Algorithme greedy amélioré pour placer les antennes.

    Stratégie :
    1. Tant qu'il reste des bâtiments non couverts
    2. Tester toutes les positions candidates (bâtiments + centres)
    3. Pour chaque position, tester tous les types d'antennes
    4. Choisir la combinaison position/type qui couvre le plus de bâtiments par euro
    5. Placer l'antenne et marquer les bâtiments comme couverts
    """
    buildings = {b["id"]: b for b in dataset["buildings"]}
    available_buildings = set(buildings.keys())
    antennas = []

    print(f"\nDébut de l'algorithme greedy amélioré...")
    print(f"Bâtiments à couvrir : {len(available_buildings)}")

    iteration = 0
    while available_buildings:
        iteration += 1
        print(f"\n--- Itération {iteration} ---")
        print(f"Bâtiments restants : {len(available_buildings)}")

        best_choice = None
        best_efficiency = 0

        # Générer les positions candidates
        candidate_positions = get_candidate_positions(buildings, available_buildings)

        # Tester chaque position candidate avec chaque type d'antenne
        for x, y, is_on_building in candidate_positions:
            for antenna_type, specs in ANTENNA_TYPES.items():
                # Trouver les bâtiments dans la portée
                in_range = get_buildings_in_range(
                    x, y, antenna_type, buildings, available_buildings
                )

                if not in_range:
                    continue

                # Vérifier la capacité
                total_pop = sum(get_max_population(buildings[bid]) for bid in in_range)
                if total_pop > specs["capacity"]:
                    # Réduire les bâtiments pour respecter la capacité
                    sorted_buildings = sorted(
                        in_range,
                        key=lambda bid: calculate_distance(
                            x, y, buildings[bid]["x"], buildings[bid]["y"]
                        ),
                    )
                    in_range = []
                    current_pop = 0
                    for bid in sorted_buildings:
                        building_pop = get_max_population(buildings[bid])
                        if current_pop + building_pop <= specs["capacity"]:
                            in_range.append(bid)
                            current_pop += building_pop

                if not in_range:
                    continue

                # Calculer le coût
                cost = (
                    specs["cost_on_building"]
                    if is_on_building
                    else specs["cost_off_building"]
                )

                # Efficacité améliorée : favorise la couverture de plus de bâtiments
                # Bonus pour couvrir beaucoup de bâtiments d'un coup
                coverage_bonus = (
                    len(in_range) ** 1.5
                )  # Favorise les grandes couvertures
                efficiency = coverage_bonus / cost

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_choice = {
                        "type": antenna_type,
                        "x": x,
                        "y": y,
                        "buildings": in_range,
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

        location_type = (
            "sur bâtiment" if best_choice["is_on_building"] else "hors bâtiment"
        )
        print(
            f"✓ Antenne {len(antennas)}: {best_choice['type']} à ({best_choice['x']}, {best_choice['y']}) [{location_type}]"
        )
        print(
            f"  Couvre {len(best_choice['buildings'])} bâtiments : {best_choice['buildings']}"
        )
        print(
            f"  Coût : {best_choice['cost']:,} € | Efficacité : {best_choice['efficiency']:.6f} bâtiments/€"
        )

        # Retirer les bâtiments couverts
        available_buildings -= set(best_choice["buildings"])

    solution = {"antennas": antennas}
    return solution


def main():
    datasets = [
        "1_peaceful_village",
        "2_small_town",
        "3_suburbia",
        "4_epitech",
        "5_isogrid",
        "6_manhattan",
    ]

    # Choisir le dataset (ou boucle sur tous)
    import sys

    if len(sys.argv) > 1:
        dataset_index = int(sys.argv[1]) - 1
        selected_datasets = [datasets[dataset_index]]
    else:
        selected_datasets = datasets[:1]  # Par défaut, le premier

    for selected_dataset in selected_datasets:
        input_file = f"./datasets/{selected_dataset}.json"

        print(f"\n{'='*70}")
        print(f"ALGORITHME GREEDY - {selected_dataset}")
        print(f"{'='*70}")

        print(f"\nChargement du dataset : {input_file}")
        dataset = json.load(open(input_file))

        print(f"\nAnalyse du dataset :")
        print(f"Nombre de bâtiments : {len(dataset['buildings'])}")

        total_pop = sum(get_max_population(b) for b in dataset["buildings"])
        print(f"Population totale : {total_pop}")

        # Génération de la solution
        solution = greedy_solution(dataset)

        print(f"\n{'='*70}")
        print(f"RÉSULTAT")
        print(f"{'='*70}")
        print(f"Solution générée avec {len(solution['antennas'])} antenne(s)")

        # Validation
        cost, isValid, message = getSolutionScore(
            json.dumps(solution), json.dumps(dataset)
        )

        print(f"\n{message}")

        if isValid:
            output_file = f"./solutions/solution_{selected_dataset}_{cost}_greedy.json"
            with open(output_file, "w") as f:
                json.dump(solution, f, indent=2)
            print(f"\n✓ Solution sauvegardée dans {output_file}")

            # Comparaison avec la solution naïve
            naive_cost = len(dataset["buildings"]) * 30_000  # Une Density par bâtiment
            improvement = ((naive_cost - cost) / naive_cost) * 100
            print(
                f"\nAmélioration vs solution naïve : {improvement:.1f}% ({naive_cost:,} € → {cost:,} €)"
            )
        else:
            print(f"\n✗ Solution invalide : {message}")

        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
