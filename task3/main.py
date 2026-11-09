"""
main.py -- Task 3 orchestrator. This file is given to you complete.

Usage:
  python main.py part1   # Part 1: Occupation by age group
  python main.py part2   # Part 2: Trips by hour of day
  python main.py all     # both (default)
"""

import sys

from task3_part1 import part1
from task3_part2 import part2


def run_all():
    part1()
    part2()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "part1":
        part1()
    elif arg == "part2":
        part2()
    elif arg == "all":
        run_all()
    else:
        print(f"Unknown argument: {arg}. Use part1, part2, or all.")
