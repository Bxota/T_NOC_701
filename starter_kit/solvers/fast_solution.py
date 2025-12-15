#!/usr/bin/env python3
"""
Solution optimisée pour les grands datasets (>1000 bâtiments).
Utilise une approche par grille pour réduire la complexité.
"""

import json
import math
import sys
import os
from collections import defaultdict
from score_function import getSolutionScore


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
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_max_population(building):
    return max(
        building["populationPeakHours"],
        building["populationOffPeakHours"],
        building["populationNight"],
    )


def grid_based_solution(dataset, grid_size=400):
    """
    Solution basée sur une grille régulière.
    Place des antennes MaxRange aux points de grille et couvre les bâtiments proches.
    """
    buildings = dataset["buildings"]

    # Trouver les limites
    min_x = min(b["x"] for b in buildings)
    max_x = max(b["x"] for b in buildings)
    min_y = min(b["y"] for b in buildings)
    max_y = max(b["y"] for b in buildings)

    print(f"Zone: X=[{min_x}, {max_x}], Y=[{min_y}, {max_y}]")

    # Créer une grille de cellules
    antennas = []
    covered = set()

    # Parcourir la grille
    y = min_y
    antenna_id = 0

    while y <= max_y + grid_size:
        x = min_x
        while x <= max_x + grid_size:
            # Trouver les bâtiments dans le rayon de cette position
            buildings_in_range = []
            total_pop = 0

            for building in buildings:
                if building["id"] in covered:
                    continue

                dist = calculate_distance(x, y, building["x"], building["y"])
                if dist <= ANTENNA_TYPES["MaxRange"]["range"]:
                    pop = get_max_population(building)
                    if total_pop + pop <= ANTENNA_TYPES["MaxRange"]["capacity"]:
                        buildings_in_range.append(building["id"])
                        total_pop += pop

            # Si on a trouvé des bâtiments, placer une antenne
            if buildings_in_range:
                antennas.append(
                    {
                        "type": "MaxRange",
                        "x": int(x),
                        "y": int(y),
                        "buildings": buildings_in_range,
                    }
                )
                covered.update(buildings_in_range)
                print(
                    f"Antenne {antenna_id}: ({int(x)}, {int(y)}) - {len(buildings_in_range)} bâtiments, {total_pop} pop"
                )
                antenna_id += 1

            x += grid_size
        y += grid_size

    print(
        f"\nPhase 1: {len(antennas)} antennes MaxRange, {len(covered)}/{len(buildings)} bâtiments couverts"
    )

    # Phase 2: Couvrir les bâtiments restants avec des antennes plus petites
    uncovered = [b for b in buildings if b["id"] not in covered]

    for building in uncovered:
        pop = get_max_population(building)

        # Choisir le type d'antenne le plus économique
        if pop <= ANTENNA_TYPES["Nano"]["capacity"]:
            antenna_type = "Nano"
        elif pop <= ANTENNA_TYPES["Spot"]["capacity"]:
            antenna_type = "Spot"
        elif pop <= ANTENNA_TYPES["Density"]["capacity"]:
            antenna_type = "Density"
        else:
            antenna_type = "MaxRange"

        antennas.append(
            {
                "type": antenna_type,
                "x": building["x"],
                "y": building["y"],
                "buildings": [building["id"]],
            }
        )
        covered.add(building["id"])

    print(f"Phase 2: +{len(uncovered)} antennes pour bâtiments isolés")
    print(
        f"Total: {len(antennas)} antennes, {len(covered)}/{len(buildings)} bâtiments couverts"
    )

    return {"antennas": antennas}


def optimized_clustering_solution(dataset):
    """
    Solution par clustering: regroupe les bâtiments proches et place des antennes optimales.
    """
    buildings = dataset["buildings"]
    antennas = []
    covered = set()

    print("Démarrage de l'algorithme de clustering...")

    # Tant qu'il reste des bâtiments à couvrir
    iteration = 0
    while len(covered) < len(buildings):
        iteration += 1
        remaining = [b for b in buildings if b["id"] not in covered]

        if iteration % 100 == 1:
            print(f"Itération {iteration}: {len(remaining)} bâtiments restants")

        # Prendre le premier bâtiment non couvert comme centre
        center = remaining[0]

        # Trouver tous les bâtiments dans un rayon de MaxRange
        cluster = []
        total_pop = 0
        max_range = ANTENNA_TYPES["MaxRange"]["range"]

        for b in remaining:
            dist = calculate_distance(center["x"], center["y"], b["x"], b["y"])
            if dist <= max_range:
                pop = get_max_population(b)
                if total_pop + pop <= ANTENNA_TYPES["MaxRange"]["capacity"]:
                    cluster.append(b["id"])
                    total_pop += pop

        # Placer une antenne MaxRange au centre
        antennas.append(
            {
                "type": "MaxRange",
                "x": center["x"],
                "y": center["y"],
                "buildings": cluster,
            }
        )

        covered.update(cluster)

    print(f"\nSolution: {len(antennas)} antennes MaxRange")
    return {"antennas": antennas}


def main():
    if len(sys.argv) > 1:
        dataset_num = sys.argv[1]
    else:
        dataset_num = "4"

    dataset_names = {
        "1": "1_peaceful_village",
        "2": "2_small_town",
        "3": "3_suburbia",
        "4": "4_epitech",
        "5": "5_isogrid",
        "6": "6_manhattan",
    }

    dataset_name = dataset_names.get(dataset_num, "4_epitech")

    # Trouver le répertoire de base
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    input_file = os.path.join(base_dir, "datasets", f"{dataset_name}.json")

    print(f"\n{'='*70}")
    print(f"SOLUTION OPTIMISÉE POUR GRANDS DATASETS")
    print(f"Dataset: {dataset_name}.json")
    print(f"{'='*70}\n")

    print("Chargement du dataset...")
    with open(input_file) as f:
        dataset = json.load(f)

    print(f"Nombre de bâtiments: {len(dataset['buildings'])}")
    total_pop = sum(get_max_population(b) for b in dataset["buildings"])
    print(f"Population totale: {total_pop:,}\n")

    # Tester les deux approches
    print("\n" + "=" * 70)
    print("MÉTHODE: Grille régulière optimisée")
    print("=" * 70)
    solution1 = grid_based_solution(dataset, grid_size=350)
    cost1, valid1, msg1 = getSolutionScore(json.dumps(solution1), json.dumps(dataset))
    print(f"\nRésultat: {msg1}")

    # Ne pas utiliser la méthode 2 (clustering) car elle est trop lente pour les grands datasets
    # solution2 = optimized_clustering_solution(dataset)

    # Utiliser seulement la solution 1
    if valid1:
        best_solution = solution1
        best_cost = cost1
        method = "grille"
    else:
        print("\n✗ Solution invalide")
        return

    print(f"\n{'='*70}")
    print(f"MEILLEURE SOLUTION: {method}")
    print(f"Coût: {best_cost:,} €")
    print(f"{'='*70}")

    # Sauvegarder
    output_dir = os.path.join(base_dir, "solutions", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"solution_{dataset_name}_{best_cost}_optimized.json"
    )
    with open(output_file, "w") as f:
        json.dump(best_solution, f, indent=2)
    print(f"\n✓ Solution sauvegardée: {output_file}")

    # Statistiques
    antenna_counts = defaultdict(int)
    for antenna in best_solution["antennas"]:
        antenna_counts[antenna["type"]] += 1

    print("\nRépartition des antennes:")
    for antenna_type, count in sorted(antenna_counts.items()):
        print(f"  - {antenna_type}: {count}")


if __name__ == "__main__":
    main()
