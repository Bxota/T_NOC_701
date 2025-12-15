#!/usr/bin/env python3
"""
Script pour tester l'algorithme greedy sur tous les datasets
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from greedy_solution import greedy_solution, get_max_population, ANTENNA_TYPES
from score_function import getSolutionScore


def test_all_datasets():
    datasets = [
        "1_peaceful_village",
        "2_small_town",
        "3_suburbia",
        "4_epitech",
        "5_isogrid",
        "6_manhattan",
    ]

    results = []

    print("\n" + "=" * 80)
    print(" " * 20 + "TEST DE L'ALGORITHME GREEDY SUR TOUS LES DATASETS")
    print("=" * 80 + "\n")

    for selected_dataset in datasets:
        input_file = f"./datasets/{selected_dataset}.json"

        try:
            dataset = json.load(open(input_file))

            print(f"📊 {selected_dataset}")
            print(f"   Bâtiments : {len(dataset['buildings'])}")

            # Génération de la solution
            solution = greedy_solution(dataset)

            # Validation
            cost, isValid, message = getSolutionScore(
                json.dumps(solution), json.dumps(dataset)
            )

            if isValid:
                output_file = (
                    f"./solutions/solution_{selected_dataset}_{cost}_greedy.json"
                )
                with open(output_file, "w") as f:
                    json.dump(solution, f, indent=2)

                naive_cost = len(dataset["buildings"]) * 30_000
                improvement = ((naive_cost - cost) / naive_cost) * 100

                results.append(
                    {
                        "dataset": selected_dataset,
                        "valid": True,
                        "cost": cost,
                        "antennas": len(solution["antennas"]),
                        "improvement": improvement,
                    }
                )

                print(f"   ✓ Coût : {cost:,} € ({len(solution['antennas'])} antennes)")
                print(f"   📈 Amélioration : {improvement:.1f}%")
            else:
                results.append(
                    {"dataset": selected_dataset, "valid": False, "error": message}
                )
                print(f"   ✗ Solution invalide : {message}")

            print()

        except Exception as e:
            print(f"   ⚠️  Erreur : {e}\n")
            results.append(
                {"dataset": selected_dataset, "valid": False, "error": str(e)}
            )

    # Récapitulatif
    print("=" * 80)
    print(" " * 30 + "RÉCAPITULATIF")
    print("=" * 80)
    print(
        f"{'Dataset':<25} {'Valide':<10} {'Coût':>15} {'Antennes':>10} {'Amélioration':>15}"
    )
    print("-" * 80)

    for result in results:
        if result["valid"]:
            print(
                f"{result['dataset']:<25} {'✓':<10} {result['cost']:>15,} € {result['antennas']:>10} {result['improvement']:>14.1f}%"
            )
        else:
            print(
                f"{result['dataset']:<25} {'✗':<10} {'N/A':>15} {'N/A':>10} {'N/A':>15}"
            )

    print("=" * 80)

    valid_results = [r for r in results if r["valid"]]
    if valid_results:
        total_cost = sum(r["cost"] for r in valid_results)
        avg_improvement = sum(r["improvement"] for r in valid_results) / len(
            valid_results
        )
        print(f"\nCoût total : {total_cost:,} €")
        print(f"Amélioration moyenne : {avg_improvement:.1f}%")
        print(f"Solutions valides : {len(valid_results)}/{len(results)}")

    print()


if __name__ == "__main__":
    test_all_datasets()
