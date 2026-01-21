"""Entry point for python -m gbxcule."""


def main() -> None:
    """Print guidance for running gbxcule."""
    print("GBxCuLE Learning Lab")
    print("====================")
    print()
    print("This package provides a GPU-native many-env Game Boy runtime.")
    print()
    print("Usage:")
    print("  uv run python -m gbxcule          # Show this help")
    print("  uv run python bench/harness.py    # Run the benchmark harness")
    print()
    print("Common commands:")
    print("  make setup    # Install dependencies via uv")
    print("  make test     # Run unit tests")
    print("  make bench    # Run baseline benchmark")
    print("  make roms     # Generate micro-ROMs")
    print("  make verify   # Run verification mode")
    print()
    print("See README.md for more information.")


if __name__ == "__main__":
    main()
