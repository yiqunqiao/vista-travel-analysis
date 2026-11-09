import sys

from task2_part1 import part1
from task2_part2 import part2
from task2_part3 import part3
from task2_evaluation import evaluation


def main():
    """This file is given to you complete -- you don't need to change it.
    It just runs whichever part(s) you ask for on the command line, e.g.
    'python main.py part2' or 'python main.py' (which runs everything)."""
    # Check whether the user typed an argument after "python main.py",
    # e.g. "part2". If not, default to running everything ('all').
    if len(sys.argv) > 1:
        arg = sys.argv[1]
    else:
        arg = 'all'

    if arg == 'part1':
        part1()
    elif arg == 'part2':
        part2()
    elif arg == 'part3':
        part3()
    elif arg == 'evaluation':
        evaluation()
    elif arg == 'all':
        # Run every part, in order.
        part1()
        part2()
        part3()
        evaluation()
    else:
        print(f"Unknown argument: {arg}. Use one of: part1, part2, part3, evaluation, all")
        sys.exit(1)


if __name__ == '__main__':
    main()
