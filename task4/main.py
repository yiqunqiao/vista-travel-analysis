import sys

from task4_setup import run_setup
from task4_part1 import part1
from task4_part2 import part2
from task4_part3 import part3


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if arg == 'setup':
        run_setup()
    elif arg == 'part1':
        part1()
    elif arg == 'part2':
        part2()
    elif arg == 'part3':
        part3()
    elif arg == 'all':
        part3()
    else:
        print(f"Unknown argument: {arg}. Use one of: setup, part1, part2, part3, all")
        sys.exit(1)


if __name__ == '__main__':
    main()
