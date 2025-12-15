#!/usr/bin/env python3

import sys
import os

# Ajouter le répertoire courant au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importer et exécuter le module
from solvers.greedy_montecarlo_solution import main

if __name__ == "__main__":
    main()
