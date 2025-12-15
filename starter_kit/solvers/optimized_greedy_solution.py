import json
import math
import random
import copy
import sys
import os
from pathlib import Path
from collections import defaultdict

# Ajouter le répertoire parent au path pour importer score_function
sys.path.insert(0, str(Path(__file__).parent.parent))
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


def save_checkpoint(solution, dataset_name, cost, tag="best_temp"):
    """Sauvegarde immédiate d'une solution dès qu'un meilleur coût est trouvé."""
    solutions_dir = Path(f"./solutions/{dataset_name}")
    solutions_dir.mkdir(parents=True, exist_ok=True)
    out_path = solutions_dir / f"solution_{dataset_name}_{cost}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(solution, f, indent=2)
    print(f"   ↳ Checkpoint sauvegardé : {out_path}")


class SpatialGrid:
    """Grille spatiale pour recherche rapide de voisins."""

    def __init__(self, buildings, cell_size=100):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        self.buildings = buildings

        # Indexer les bâtiments dans la grille
        for bid, building in buildings.items():
            cell = self._get_cell(building["x"], building["y"])
            self.grid[cell].append(bid)

    def _get_cell(self, x, y):
        """Retourne la cellule de grille pour une position."""
        return (x // self.cell_size, y // self.cell_size)

    def get_neighbors(self, x, y, radius):
        """Retourne tous les bâtiments dans un rayon donné."""
        # Déterminer les cellules à vérifier
        cell_radius = math.ceil(radius / self.cell_size)
        center_cell = self._get_cell(x, y)

        candidates = []
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                candidates.extend(self.grid.get(cell, []))

        # Filtrer par distance exacte
        result = []
        radius_sq = radius * radius
        for bid in candidates:
            building = self.buildings[bid]
            dist_sq = (x - building["x"]) ** 2 + (y - building["y"]) ** 2
            if dist_sq <= radius_sq:
                result.append(bid)

        return result


class DistanceCache:
    """Cache des distances précalculées."""

    def __init__(self):
        self.cache = {}

    def get_distance(self, x1, y1, x2, y2):
        """Retourne la distance avec mise en cache."""
        key = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if key not in self.cache:
            self.cache[key] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return self.cache[key]


def get_max_population(building):
    """Retourne la population maximale d'un bâtiment sur les 3 périodes."""
    return max(
        building["populationPeakHours"],
        building["populationOffPeakHours"],
        building["populationNight"],
    )


def cluster_buildings(buildings_dict, available_buildings, max_distance=150):
    """
    Regroupe les bâtiments proches en clusters pour un traitement plus efficace.
    """
    clusters = []
    remaining = available_buildings.copy()

    while remaining:
        # Prendre un bâtiment aléatoire comme seed
        seed = next(iter(remaining))
        cluster = {seed}
        remaining.discard(seed)

        # Ajouter tous les bâtiments proches
        to_check = {seed}
        while to_check:
            current = to_check.pop()
            current_building = buildings_dict[current]

            for bid in list(remaining):
                building = buildings_dict[bid]
                dist = math.sqrt(
                    (current_building["x"] - building["x"]) ** 2
                    + (current_building["y"] - building["y"]) ** 2
                )
                if dist <= max_distance:
                    cluster.add(bid)
                    remaining.discard(bid)
                    to_check.add(bid)

        clusters.append(cluster)

    return clusters


def get_strategic_positions(buildings_dict, available_buildings, grid):
    """
    Génère des positions candidates stratégiques pour les antennes.
    """
    positions = []

    # 1. Sur chaque bâtiment (bonus de coût)
    for bid in available_buildings:
        building = buildings_dict[bid]
        positions.append((building["x"], building["y"], True, bid))

    # 2. Barycentres de clusters de bâtiments proches
    if len(available_buildings) > 5:
        # K-means simplifié pour trouver des centres
        n_centers = min(10, len(available_buildings) // 5)
        centers = random.sample(list(available_buildings), n_centers)

        for _ in range(3):  # 3 itérations de k-means
            clusters_map = defaultdict(list)

            # Assigner chaque bâtiment au centre le plus proche
            for bid in available_buildings:
                building = buildings_dict[bid]
                min_dist = float("inf")
                closest_center = centers[0]

                for center_id in centers:
                    center = buildings_dict[center_id]
                    dist = (building["x"] - center["x"]) ** 2 + (
                        building["y"] - center["y"]
                    ) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        closest_center = center_id

                clusters_map[closest_center].append(bid)

            # Recalculer les centres
            new_centers = []
            for cluster_bids in clusters_map.values():
                if cluster_bids:
                    avg_x = sum(
                        buildings_dict[bid]["x"] for bid in cluster_bids
                    ) // len(cluster_bids)
                    avg_y = sum(
                        buildings_dict[bid]["y"] for bid in cluster_bids
                    ) // len(cluster_bids)

                    # Trouver le bâtiment le plus proche de ce centre
                    min_dist = float("inf")
                    closest = cluster_bids[0]
                    for bid in cluster_bids:
                        building = buildings_dict[bid]
                        dist = (building["x"] - avg_x) ** 2 + (
                            building["y"] - avg_y
                        ) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            closest = bid
                    new_centers.append(closest)

            if new_centers:
                centers = new_centers

        # Ajouter les centres comme positions candidates
        for center_id in centers:
            center = buildings_dict[center_id]
            positions.append((center["x"], center["y"], True, center_id))

    return positions


def select_buildings_by_priority(
    x, y, antenna_type, buildings_dict, candidates, distance_cache
):
    """
    Sélectionne les bâtiments selon une priorité intelligente.
    Priorité : population élevée + proximité
    """
    specs = ANTENNA_TYPES[antenna_type]

    # Calculer un score pour chaque bâtiment
    scored_buildings = []
    for bid in candidates:
        building = buildings_dict[bid]
        pop = get_max_population(building)
        dist = distance_cache.get_distance(x, y, building["x"], building["y"])

        # Score : favorise population élevée et proximité
        # Plus le bâtiment est proche et peuplé, plus le score est élevé
        score = pop / (1 + dist / specs["range"])
        scored_buildings.append((bid, pop, score))

    # Trier par score décroissant
    scored_buildings.sort(key=lambda x: x[2], reverse=True)

    # Sélectionner jusqu'à la capacité
    selected = []
    current_pop = 0

    for bid, pop, score in scored_buildings:
        if current_pop + pop <= specs["capacity"]:
            selected.append(bid)
            current_pop += pop
        elif current_pop == 0 and pop <= specs["capacity"]:
            # Si c'est le premier et qu'il rentre seul dans la capacité
            selected.append(bid)
            break

    return selected


def greedy_optimized(
    buildings_dict, available_buildings, grid, distance_cache, efficiency_func=None
):
    """
    Algorithme greedy optimisé avec structures de données spatiales.
    """
    available = available_buildings.copy()
    antennas = []

    # Fonction d'efficacité par défaut
    if efficiency_func is None:
        efficiency_func = (
            lambda n_buildings, cost, total_pop: (n_buildings**1.5 * total_pop**0.5)
            / cost
        )

    iteration = 0
    while available:
        iteration += 1
        best_choice = None
        best_efficiency = 0

        # Positions candidates stratégiques
        positions = get_strategic_positions(buildings_dict, available, grid)

        # Limiter le nombre de positions à tester pour les gros datasets
        if len(positions) > 100:
            # Échantillonner les positions : garder tous les bâtiments + quelques centres
            building_positions = [p for p in positions if p[2]]  # Sur bâtiment
            other_positions = [p for p in positions if not p[2]]
            if len(other_positions) > 20:
                other_positions = random.sample(other_positions, 20)
            positions = building_positions + other_positions

        # Types d'antennes à tester en priorité selon la situation
        if len(available) > 100:
            # Pour beaucoup de bâtiments, favoriser les grandes antennes
            antenna_priority = ["MaxRange", "Density", "Spot", "Nano"]
        elif len(available) > 20:
            antenna_priority = ["Density", "MaxRange", "Spot", "Nano"]
        elif len(available) > 5:
            antenna_priority = ["Spot", "Density", "Nano", "MaxRange"]
        else:
            # Pour les derniers bâtiments, tester toutes les options
            antenna_priority = ["Nano", "Spot", "Density", "MaxRange"]

        # Tester chaque combinaison position × type
        for x, y, is_on_building, _ in positions:
            for antenna_type in antenna_priority:
                specs = ANTENNA_TYPES[antenna_type]

                # Trouver les bâtiments dans la portée (utilise la grille spatiale)
                in_range = [
                    bid
                    for bid in grid.get_neighbors(x, y, specs["range"])
                    if bid in available
                ]

                if not in_range:
                    continue

                # Sélectionner les bâtiments selon la capacité
                selected = select_buildings_by_priority(
                    x, y, antenna_type, buildings_dict, in_range, distance_cache
                )

                if not selected:
                    continue

                # Calculer le coût
                cost = (
                    specs["cost_on_building"]
                    if is_on_building
                    else specs["cost_off_building"]
                )

                # Calculer la population totale couverte
                total_pop = sum(
                    get_max_population(buildings_dict[bid]) for bid in selected
                )

                # Calculer l'efficacité
                efficiency = efficiency_func(len(selected), cost, total_pop)

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_choice = {
                        "type": antenna_type,
                        "x": x,
                        "y": y,
                        "buildings": selected,
                        "cost": cost,
                        "efficiency": efficiency,
                    }

        if not best_choice:
            print(f"⚠️  Impossible de couvrir {len(available)} bâtiments restants!")
            # Essayer de couvrir avec des antennes Nano individuelles
            for bid in available:
                building = buildings_dict[bid]
                antennas.append(
                    {
                        "type": "Nano",
                        "x": building["x"],
                        "y": building["y"],
                        "buildings": [bid],
                    }
                )
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

        if iteration % 50 == 0:
            print(
                f"  Itération {iteration}: {len(available)} bâtiments restants, {len(antennas)} antennes placées"
            )

    return antennas


def monte_carlo_optimized(dataset, dataset_name, n_iterations=100):
    """
    Monte Carlo optimisé avec structures de données spatiales.
    """
    buildings = {b["id"]: b for b in dataset["buildings"]}
    available_buildings = set(buildings.keys())

    # Créer les structures de données optimisées
    print("Préparation des structures de données...")
    grid = SpatialGrid(buildings, cell_size=100)
    distance_cache = DistanceCache()

    best_solution = None
    best_cost = float("inf")

    print(f"\n{'='*70}")
    print(f"MONTE CARLO OPTIMISÉ")
    print(f"{'='*70}")
    print(f"Bâtiments à couvrir : {len(available_buildings)}")
    print(f"Nombre d'itérations : {n_iterations}")

    # Différentes fonctions d'efficacité
    efficiency_functions = [
        # Standard avec bonus population
        lambda n, c, p: (n**1.5 * p**0.5) / c,
        # Favorise grandes couvertures
        lambda n, c, p: (n**2 * p**0.3) / c,
        # Favorise population
        lambda n, c, p: (n**1.2 * p**0.8) / c,
        # Simple ratio
        lambda n, c, p: (n * p) / c,
        # Maximise nombre de bâtiments
        lambda n, c, p: n**1.8 / c,
        # Équilibre
        lambda n, c, p: (n * p**0.5) / (c**0.9),
    ]

    for iteration in range(n_iterations):
        # Choisir une fonction aléatoirement
        efficiency_func = random.choice(efficiency_functions)

        # Ajouter du bruit aléatoire (20%)
        original_func = efficiency_func
        efficiency_func = lambda n, c, p, of=original_func: of(
            n, c, p
        ) * random.uniform(0.85, 1.15)

        # Générer une solution
        antennas = greedy_optimized(
            buildings, available_buildings, grid, distance_cache, efficiency_func
        )

        # Calculer le coût
        solution = {"antennas": antennas}
        cost, is_valid, message = getSolutionScore(
            json.dumps(solution), json.dumps(dataset)
        )

        if is_valid and cost < best_cost:
            best_cost = cost
            best_solution = solution
            print(
                f"🎯 Itération {iteration + 1}/{n_iterations}: Nouveau record ! Coût = {cost:,} € ({len(antennas)} antennes)"
            )
            save_checkpoint(best_solution, dataset_name, best_cost, tag="best_temp")
        elif iteration % 5 == 0:
            status = "✓" if is_valid else "✗"
            print(
                f"   Itération {iteration + 1}/{n_iterations}: {status} Coût = {cost:,} € ({len(antennas)} antennes)"
            )

    return best_solution


def post_optimize(solution, dataset):
    """
    Post-optimisation : essayer de remplacer des antennes par des types moins chers.
    """
    print(f"\n{'='*70}")
    print(f"POST-OPTIMISATION")
    print(f"{'='*70}")

    buildings = {b["id"]: b for b in dataset["buildings"]}
    current = copy.deepcopy(solution)

    cost, _, _ = getSolutionScore(json.dumps(current), json.dumps(dataset))
    initial_cost = cost

    improved = True
    iteration = 0

    while improved and iteration < 10:
        improved = False
        iteration += 1

        for i, antenna in enumerate(current["antennas"]):
            # Essayer des types moins chers
            current_type = antenna["type"]
            current_cost_on = ANTENNA_TYPES[current_type]["cost_on_building"]
            current_cost_off = ANTENNA_TYPES[current_type]["cost_off_building"]

            # Trouver si l'antenne est sur un bâtiment
            is_on_building = (antenna["x"], antenna["y"]) in [
                (buildings[bid]["x"], buildings[bid]["y"])
                for bid in antenna["buildings"]
            ]

            current_antenna_cost = (
                current_cost_on if is_on_building else current_cost_off
            )

            for new_type in ANTENNA_TYPES.keys():
                if new_type == current_type:
                    continue

                new_specs = ANTENNA_TYPES[new_type]
                new_cost = (
                    new_specs["cost_on_building"]
                    if is_on_building
                    else new_specs["cost_off_building"]
                )

                # Ne tester que si moins cher
                if new_cost >= current_antenna_cost:
                    continue

                # Vérifier si le nouveau type peut couvrir les mêmes bâtiments
                can_cover = True
                total_pop = 0

                for bid in antenna["buildings"]:
                    building = buildings[bid]
                    dist = math.sqrt(
                        (antenna["x"] - building["x"]) ** 2
                        + (antenna["y"] - building["y"]) ** 2
                    )

                    if dist > new_specs["range"]:
                        can_cover = False
                        break

                    total_pop += get_max_population(building)

                if not can_cover or total_pop > new_specs["capacity"]:
                    continue

                # Tester le changement
                test_solution = copy.deepcopy(current)
                test_solution["antennas"][i]["type"] = new_type

                new_cost_total, is_valid, _ = getSolutionScore(
                    json.dumps(test_solution), json.dumps(dataset)
                )

                if is_valid and new_cost_total < cost:
                    print(
                        f"  ✓ Antenne {i}: {current_type} → {new_type} (gain: {cost - new_cost_total:,} €)"
                    )
                    current = test_solution
                    cost = new_cost_total
                    improved = True
                    break

    print(f"\nCôut initial: {initial_cost:,} €")
    print(f"Coût final: {cost:,} €")
    print(
        f"Gain: {initial_cost - cost:,} € ({(initial_cost - cost) / initial_cost * 100:.2f}%)"
    )

    return current


def main():
    import sys

    if len(sys.argv) > 1:
        dataset_num = sys.argv[1]
    else:
        dataset_num = "3"

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
    print(f"SOLUTION GREEDY OPTIMISÉE")
    print(f"Dataset : {dataset_name}.json")
    print(f"{'='*70}")

    dataset = json.load(open(input_file))

    print(f"\nAnalyse du dataset :")
    print(f"Nombre de bâtiments : {len(dataset['buildings'])}")
    total_pop = sum(get_max_population(b) for b in dataset["buildings"])
    print(f"Population totale : {total_pop:,}")

    # Forcer un run rapide : seulement 5 itérations pour tous les datasets
    n_iterations = 5
    print(f"Nombre d'itérations : {n_iterations} (run rapide)")

    # Phase Monte Carlo
    solution = monte_carlo_optimized(dataset, dataset_name, n_iterations=n_iterations)

    if solution:
        # Phase post-optimisation
        solution = post_optimize(solution, dataset)

        cost, is_valid, message = getSolutionScore(
            json.dumps(solution), json.dumps(dataset)
        )

        print(f"\n{'='*70}")
        print(f"RÉSULTAT FINAL")
        print(f"{'='*70}")
        print(f"{message}")

        if is_valid:
            output_file = f"./solutions/{dataset_name}/solution_{dataset_name}_{cost}_optimized.json"
            with open(output_file, "w") as f:
                json.dump(solution, f, indent=2)
            print(f"\n✓ Solution sauvegardée dans {output_file}")

            # Statistiques
            print(f"\nDétails de la solution :")
            antenna_counts = {}
            for antenna in solution["antennas"]:
                antenna_type = antenna["type"]
                antenna_counts[antenna_type] = antenna_counts.get(antenna_type, 0) + 1

            for antenna_type, count in sorted(antenna_counts.items()):
                print(f"  - {antenna_type}: {count} antenne(s)")

            naive_cost = len(dataset["buildings"]) * 30_000
            improvement = ((naive_cost - cost) / naive_cost) * 100
            print(f"\nAmélioration vs solution naïve : {improvement:.1f}%")
        else:
            print(f"\n✗ Solution invalide : {message}")
    else:
        print("\n✗ Aucune solution trouvée")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
