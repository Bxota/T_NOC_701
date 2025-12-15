#!/usr/bin/env python3
"""
Script pour comparer les différentes solutions générées.
"""

import json
import os
from pathlib import Path
from score_function import getSolutionScore


def analyze_solution(solution_file, dataset_file):
    """Analyse une solution et retourne ses stats."""
    with open(solution_file) as f:
        solution = json.load(f)

    with open(dataset_file) as f:
        dataset = json.load(f)

    cost, is_valid, message = getSolutionScore(
        json.dumps(solution), json.dumps(dataset)
    )

    # Compter les antennes par type
    antenna_counts = {}
    for antenna in solution.get("antennas", []):
        antenna_type = antenna["type"]
        antenna_counts[antenna_type] = antenna_counts.get(antenna_type, 0) + 1

    # Compter les bâtiments couverts
    buildings_covered = set()
    for antenna in solution.get("antennas", []):
        buildings_covered.update(antenna.get("buildings", []))

    return {
        "cost": cost,
        "is_valid": is_valid,
        "message": message,
        "n_antennas": len(solution.get("antennas", [])),
        "antenna_counts": antenna_counts,
        "buildings_covered": len(buildings_covered),
        "buildings_total": len(dataset["buildings"]),
    }


def main():
    datasets = {
        "1_peaceful_village": "./datasets/1_peaceful_village.json",
        "2_small_town": "./datasets/2_small_town.json",
        "3_suburbia": "./datasets/3_suburbia.json",
        "4_epitech": "./datasets/4_epitech.json",
        "5_isogrid": "./datasets/5_isogrid.json",
        "6_manhattan": "./datasets/6_manhattan.json",
    }

    print("\n" + "=" * 100)
    print("COMPARAISON DES SOLUTIONS")
    print("=" * 100)

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 100)

        solutions_dir = Path(f"./solutions/{dataset_name}")
        if not solutions_dir.exists():
            print("  Aucune solution trouvée")
            continue

        solutions = list(solutions_dir.glob("solution_*.json"))
        if not solutions:
            print("  Aucune solution trouvée")
            continue

        # Analyser toutes les solutions
        results = []
        for solution_file in solutions:
            try:
                stats = analyze_solution(solution_file, dataset_path)
                stats["filename"] = solution_file.name
                results.append(stats)
            except Exception as e:
                print(f"  ⚠️  Erreur avec {solution_file.name}: {e}")

        if not results:
            continue

        # Trier par coût
        results.sort(key=lambda x: x["cost"] if x["is_valid"] else float("inf"))

        # Afficher les résultats
        print(
            f"{'Fichier':<50} {'Coût':>15} {'Antennes':>10} {'Couverture':>12} {'Statut':>10}"
        )
        print("-" * 100)

        for result in results:
            status = "✓ VALID" if result["is_valid"] else "✗ INVALID"
            cost_str = f"{result['cost']:,} €" if result["is_valid"] else "N/A"
            coverage = f"{result['buildings_covered']}/{result['buildings_total']}"

            print(
                f"{result['filename']:<50} {cost_str:>15} {result['n_antennas']:>10} {coverage:>12} {status:>10}"
            )

        # Afficher la meilleure solution
        best = results[0]
        if best["is_valid"]:
            print(f"\n  🏆 Meilleure solution: {best['filename']}")
            print(f"     Coût: {best['cost']:,} €")
            print(f"     Antennes: {best['n_antennas']}")
            for antenna_type, count in sorted(best["antenna_counts"].items()):
                print(f"       - {antenna_type}: {count}")

    print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
