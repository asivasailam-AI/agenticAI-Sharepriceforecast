import pickle
import sys
from importlib.metadata import PackageNotFoundError, version


LIBRARIES = (
    ("Flask", "Flask"),
    ("NumPy", "numpy"),
    ("Pandas", "pandas"),
    ("Scikit-learn", "scikit-learn"),
    ("Joblib", "joblib"),
    ("nselib", "nselib"),
    ("Statsmodels", "statsmodels"),
)


def print_library_versions():
    print("=" * 50)
    print("Python Environment Information")
    print("=" * 50)

    print(f"Python         : {sys.version}")
    print(f"Pickle         : {pickle.format_version}")

    for label, distribution in LIBRARIES:
        try:
            library_version = version(distribution)
        except PackageNotFoundError:
            library_version = "not installed"

        print(f"{label:<15}: {library_version}")

    print("=" * 50)


if __name__ == "__main__":
    print_library_versions()
