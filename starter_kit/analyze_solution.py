#!/usr/bin/env python3
"""
Analyse détaillée d'une solution pour comprendre la répartition des antennes.
"""

import json
import sys
import math
from collections import defaultdict


def analyze_detailed(solution_file, dataset_file):
    """Analyse détaillée d'une solution."""

    with open(solution_file) as f:
        solution = json.load(f)

    with open(dataset_file) as f:
        dataset = json.load(f)

    buildings = {b["id"]: b for b in dataset["buildings"]}

    print("\n" + "=" * 80)
    print(f"ANALYSE DÉTAILLÉE")
    print("=" * 80)
    print(f"Solution : {solution_file}")
    print(f"Dataset  : {dataset_file}")
    print("=" * 80)

    # Stats globales
    n_antennas = len(solution["antennas"])
    n_buildings = len(buildings)

    print(f"\n📊 STATISTIQUES GLOBALES")
    print(f"   Antennes installées : {n_antennas}")
    print(f"   Bâtiments à couvrir : {n_buildings}")
    print(f"   Ratio : {n_buildings / n_antennas:.2f} bâtiments/antenne")

    # Stats par type d'antenne
    print(f"\n📡 RÉPARTITION PAR TYPE")
    antenna_stats = defaultdict(
        lambda: {"count": 0, "cost": 0, "buildings": 0, "capacity_used": []}
    )

    total_cost = 0

    for antenna in solution["antennas"]:
        antenna_type = antenna["type"]

        # Déterminer si sur bâtiment
        is_on_building = (antenna["x"], antenna["y"]) in [
            (buildings[bid]["x"], buildings[bid]["y"])
            for bid in antenna.get("buildings", [])
        ]

        # Coût
        if antenna_type == "Nano":
            cost = 5000 if is_on_building else 6000
            capacity = 200
            range_m = 50
        elif antenna_type == "Spot":
            cost = 15000 if is_on_building else 20000
            capacity = 800
            range_m = 100
        elif antenna_type == "Density":
            cost = 30000 if is_on_building else 50000
            capacity = 5000
            range_m = 150
        else:  # MaxRange
            cost = 40000 if is_on_building else 50000
            capacity = 3500
            range_m = 400

        total_cost += cost

        # Population couverte
        pop = 0
        for bid in antenna.get("buildings", []):
            building = buildings[bid]
            pop += max(
                building["populationPeakHours"],
                building["populationOffPeakHours"],
                building["populationNight"],
            )

        capacity_percent = (pop / capacity * 100) if capacity > 0 else 0

        antenna_stats[antenna_type]["count"] += 1
        antenna_stats[antenna_type]["cost"] += cost
        antenna_stats[antenna_type]["buildings"] += len(antenna.get("buildings", []))
        antenna_stats[antenna_type]["capacity_used"].append(capacity_percent)

    for antenna_type in ["Nano", "Spot", "Density", "MaxRange"]:
        if antenna_type not in antenna_stats:
            continue

        stats = antenna_stats[antenna_type]
        count = stats["count"]
        avg_buildings = stats["buildings"] / count
        avg_capacity = sum(stats["capacity_used"]) / len(stats["capacity_used"])

        print(f"\n   {antenna_type:8} : {count:4} antennes")
        print(f"              Coût total     : {stats['cost']:,} €")
        print(f"              Bâtiments/ant. : {avg_buildings:.2f}")
        print(f"              Capacité util. : {avg_capacity:.1f}%")

    print(f"\n💰 COÛT TOTAL : {total_cost:,} €")

    # Distribution des bâtiments par antenne
    print(f"\n📈 DISTRIBUTION DES BÂTIMENTS PAR ANTENNE")
    buildings_per_antenna = [len(a.get("buildings", [])) for a in solution["antennas"]]
    buildings_per_antenna.sort(reverse=True)

    print(f"   Maximum   : {max(buildings_per_antenna)} bâtiments")
    print(f"   Minimum   : {min(buildings_per_antenna)} bâtiments")
    print(
        f"   Moyenne   : {sum(buildings_per_antenna) / len(buildings_per_antenna):.2f} bâtiments"
    )
    print(
        f"   Médiane   : {buildings_per_antenna[len(buildings_per_antenna) // 2]} bâtiments"
    )

    # Histogramme simplifié
    print(f"\n   Histogramme (nombre d'antennes par tranche) :")
    ranges = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100)]
    for min_b, max_b in ranges:
        count = sum(1 for x in buildings_per_antenna if min_b <= x <= max_b)
        if count > 0:
            bar = "█" * (count // max(1, n_antennas // 50))
            label = f"{min_b}" if min_b == max_b else f"{min_b}-{max_b}"
            print(f"   {label:8} bât.: {count:4} {bar}")

    # Population couverte
    print(f"\n👥 POPULATION")
    total_pop = sum(
        max(b["populationPeakHours"], b["populationOffPeakHours"], b["populationNight"])
        for b in buildings.values()
    )
    print(f"   Population totale : {total_pop:,} habitants")
    print(f"   Coût par habitant : {total_cost / total_pop:.2f} €/hab")

    # Densité spatiale
    print(f"\n🗺️  DENSITÉ SPATIALE")
    if buildings:
        x_coords = [b["x"] for b in buildings.values()]
        y_coords = [b["y"] for b in buildings.values()]
        area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

        print(f"   Zone couverte     : {area:,} unités²")
        print(f"   Densité bâtiments : {n_buildings / area * 1000:.2f} bât./1000u²")
        print(f"   Densité antennes  : {n_antennas / area * 1000:.2f} ant./1000u²")

    print("\n" + "=" * 80 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_solution.py <solution_file> [dataset_file]")
        print("\nExemple:")
        print(
            "  python3 analyze_solution.py solutions/3_suburbia/solution_3_suburbia_12345_optimized.json"
        )
        sys.exit(1)

    solution_file = sys.argv[1]

    # Déduire le dataset du nom du fichier solution
    if len(sys.argv) >= 3:
        dataset_file = sys.argv[2]
    else:
        # Extraire le nom du dataset du fichier solution
        import re

        match = re.search(r"(\d+_\w+)", solution_file)
        if match:
            dataset_name = match.group(1)
            dataset_file = f"datasets/{dataset_name}.json"
        else:
            print("❌ Impossible de déduire le dataset. Spécifiez-le en argument.")
            sys.exit(1)

    try:
        analyze_detailed(solution_file, dataset_file)
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé : {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
