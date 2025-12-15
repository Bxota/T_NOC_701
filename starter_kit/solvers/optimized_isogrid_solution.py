import json
import math
from pathlib import Path
from collections import defaultdict

# Réutiliser les briques avancées existantes
from solvers.optimized_greedy_solution import (
    ANTENNA_TYPES,
    SpatialGrid,
    DistanceCache,
    get_max_population,
    greedy_optimized,
    post_optimize,
)
from score_function import getSolutionScore


def build_density_cells(buildings, available_buildings, cell_size=250):
    """Construit une carte de densité (population max) par cellule."""
    cells = defaultdict(lambda: {"pop": 0, "bids": []})
    for bid in available_buildings:
        b = buildings[bid]
        cx, cy = b["x"] // cell_size, b["y"] // cell_size
        pop = get_max_population(b)
        cells[(cx, cy)]["pop"] += pop
        cells[(cx, cy)]["bids"].append(bid)
    return cells


def nearest_building_to(x, y, buildings_dict, candidate_bids):
    """Retourne l'id du bâtiment le plus proche d'un point parmi la liste."""
    best = None
    best_d2 = float("inf")
    for bid in candidate_bids:
        b = buildings_dict[bid]
        d2 = (x - b["x"]) ** 2 + (y - b["y"]) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = bid
    return best


def select_with_capacity_by_distance(
    x, y, antenna_type, buildings_dict, candidate_bids
):
    """Sélectionne des bâtiments proches en respectant la capacité, triés par distance."""
    specs = ANTENNA_TYPES[antenna_type]
    r2 = specs["range"] * specs["range"]
    in_range = []
    for bid in candidate_bids:
        b = buildings_dict[bid]
        d2 = (x - b["x"]) ** 2 + (y - b["y"]) ** 2
        if d2 <= r2:
            in_range.append((bid, d2))
    if not in_range:
        return []
    in_range.sort(key=lambda t: t[1])

    selected = []
    total = 0
    for bid, _ in in_range:
        pop = get_max_population(buildings_dict[bid])
        if total + pop <= specs["capacity"]:
            selected.append(bid)
            total += pop
    return selected


def density_guided_cover(dataset, cell_size=250, top_cells_ratio=0.25):
    """
    Couverture guidée par densité pour grilles régulières (dataset 5_isogrid).
    - Binning spatial par cellules
    - Ancrage sur le bâtiment le plus proche du centroïde (coût on_building)
    - Choix entre MaxRange et Density selon couverture effective/cout
    - Marque les bâtiments couverts puis itère
    Retourne une liste d'antennes et l'ensemble des bâtiments couverts.
    """
    buildings = {b["id"]: b for b in dataset["buildings"]}
    available = set(buildings.keys())
    antennas = []

    grid = SpatialGrid(buildings, cell_size=max(150, cell_size))
    dist_cache = DistanceCache()

    iteration = 0
    while available:
        iteration += 1
        cells = build_density_cells(buildings, available, cell_size)
        if not cells:
            break

        ordered = sorted(cells.items(), key=lambda kv: kv[1]["pop"], reverse=True)
        limit = max(1, int(len(ordered) * top_cells_ratio))
        batch = ordered[:limit]

        progressed = False
        for (cx, cy), payload in batch:
            bids = [bid for bid in payload["bids"] if bid in available]
            if not bids:
                continue

            avg_x = sum(buildings[bid]["x"] for bid in bids) / len(bids)
            avg_y = sum(buildings[bid]["y"] for bid in bids) / len(bids)
            anchor_bid = nearest_building_to(avg_x, avg_y, buildings, bids)
            anchor = buildings[anchor_bid]
            ax, ay = anchor["x"], anchor["y"]

            best_choice = None
            best_score = -1

            for a_type in ("MaxRange", "Density"):
                neighbors = [
                    bid
                    for bid in grid.get_neighbors(
                        ax, ay, ANTENNA_TYPES[a_type]["range"]
                    )
                    if bid in available
                ]
                if not neighbors:
                    continue

                selected = select_with_capacity_by_distance(
                    ax, ay, a_type, buildings, neighbors
                )
                if not selected:
                    continue

                total_pop = sum(get_max_population(buildings[bid]) for bid in selected)
                cost_on = ANTENNA_TYPES[a_type]["cost_on_building"]
                score = (len(selected) ** 1.3) * (total_pop**0.7) / cost_on

                if score > best_score:
                    best_score = score
                    best_choice = {
                        "type": a_type,
                        "x": ax,
                        "y": ay,
                        "buildings": selected,
                    }

            if best_choice is None:
                continue

            antennas.append(best_choice)
            available -= set(best_choice["buildings"])
            progressed = True

        if not progressed:
            break

        if iteration >= 50:
            break

    covered = set(bid for a in antennas for bid in a["buildings"])
    return antennas, covered


def solve_isogrid(dataset):
    """Pipeline spécialisé pour 5_isogrid: densité puis greedy optimisé, puis post-optimisation."""
    antennas_dens, covered = density_guided_cover(
        dataset, cell_size=250, top_cells_ratio=0.25
    )

    if len(covered) < len(dataset["buildings"]):
        buildings = {b["id"]: b for b in dataset["buildings"]}
        grid = SpatialGrid(buildings, cell_size=100)
        distance_cache = DistanceCache()
        remaining = set(b["id"] for b in dataset["buildings"]) - covered
        antennas_tail = greedy_optimized(buildings, remaining, grid, distance_cache)
        antennas = antennas_dens + antennas_tail
    else:
        antennas = antennas_dens

    solution = {"antennas": antennas}
    solution = post_optimize(solution, dataset)
    return solution


def main():
    import sys

    if len(sys.argv) > 1:
        dataset_num = sys.argv[1]
    else:
        dataset_num = "5"

    dataset_names = {
        "1": "1_peaceful_village",
        "2": "2_small_town",
        "3": "3_suburbia",
        "4": "4_epitech",
        "5": "5_isogrid",
        "6": "6_manhattan",
    }

    dataset_name = dataset_names.get(dataset_num, "5_isogrid")
    input_file = f"./datasets/{dataset_name}.json"

    print(f"\n{'='*70}")
    print(f"SOLUTION SPÉCIALISÉE ISOGRID")
    print(f"Dataset : {dataset_name}.json")
    print(f"{'='*70}")

    dataset = json.load(open(input_file))

    print(f"\nAnalyse du dataset :")
    print(f"Nombre de bâtiments : {len(dataset['buildings'])}")
    total_pop = sum(get_max_population(b) for b in dataset["buildings"])
    print(f"Population totale : {total_pop:,}")

    solution = solve_isogrid(dataset)

    cost, is_valid, message = getSolutionScore(
        json.dumps(solution), json.dumps(dataset)
    )

    print(f"\n{'='*70}")
    print(f"RÉSULTAT FINAL (ISOGRID)")
    print(f"{'='*70}")
    print(f"{message}")

    if is_valid:
        out_dir = Path(f"./solutions/{dataset_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"solution_{dataset_name}_{cost}_optimized.json"
        with open(out_path, "w") as f:
            json.dump(solution, f, indent=2)
        print(f"\n✓ Solution sauvegardée dans {out_path}")
    else:
        print(f"\n✗ Solution invalide : {message}")


if __name__ == "__main__":
    main()
