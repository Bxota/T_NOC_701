import json
import math
from score_function import getSolutionScore


def optimized_solution_peaceful_village(dataset):
    """
    Solution optimisée pour peaceful_village :
    - 5 bâtiments alignés horizontalement
    - Une seule antenne MaxRange au centre peut tous les couvrir
    """
    # Tous les bâtiments sont sur y=0, de x=0 à x=400
    # Une antenne MaxRange (portée 400) à (200, 0) couvre tous les bâtiments
    # Population totale : 250+100+420+120+60 = 950 < 3500 (capacité MaxRange)

    solution = {
        "antennas": [
            {"type": "MaxRange", "x": 200, "y": 0, "buildings": [0, 1, 2, 3, 4]}
        ]
    }

    return solution


def calculate_distance(x1, y1, x2, y2):
    """Calcule la distance euclidienne entre deux points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def main():
    input_file = "./datasets/1_peaceful_village.json"

    print(f"Chargement du dataset : {input_file}")
    dataset = json.load(open(input_file))

    print(f"\nAnalyse du dataset :")
    print(f"Nombre de bâtiments : {len(dataset['buildings'])}")

    # Analyse des bâtiments
    total_pop = 0
    for building in dataset["buildings"]:
        max_pop = max(
            building["populationPeakHours"],
            building["populationOffPeakHours"],
            building["populationNight"],
        )
        total_pop += max_pop
        print(
            f"  Bâtiment {building['id']}: ({building['x']}, {building['y']}) - pop max: {max_pop}"
        )

    print(f"\nPopulation totale : {total_pop}")

    print("\n" + "=" * 60)
    print("Génération de la solution optimisée...")
    solution = optimized_solution_peaceful_village(dataset)

    print(f"Solution générée avec {len(solution['antennas'])} antenne(s)")

    # Affichage des détails de la solution
    for i, antenna in enumerate(solution["antennas"]):
        print(f"\nAntenne {i} :")
        print(f"  Type: {antenna['type']}")
        print(f"  Position: ({antenna['x']}, {antenna['y']})")
        print(f"  Bâtiments couverts: {antenna['buildings']}")

        # Vérification des distances
        for building_id in antenna["buildings"]:
            building = dataset["buildings"][building_id]
            dist = calculate_distance(
                antenna["x"], antenna["y"], building["x"], building["y"]
            )
            print(f"    - Bâtiment {building_id}: distance = {dist:.2f}")

    print("\n" + "=" * 60)
    print("Validation de la solution...")
    cost, isValid, message = getSolutionScore(json.dumps(solution), json.dumps(dataset))

    print(f"\n{message}")
    print(f"Valide: {isValid}")
    print(f"Coût: {cost:,} €")

    if isValid:
        output_file = f"./solutions/solution_1_peaceful_village_{cost}.json"
        with open(output_file, "w") as f:
            json.dump(solution, f, indent=2)
        print(f"\n✓ Solution sauvegardée dans {output_file}")

    return isValid, cost


if __name__ == "__main__":
    main()
