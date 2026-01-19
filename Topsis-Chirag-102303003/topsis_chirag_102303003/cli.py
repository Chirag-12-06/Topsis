import sys
from .topsis import topsis, TopsisError


def main():
    if len(sys.argv) != 5:
        print("Usage: topsis <InputDataFile> <Weights> <Impacts> <OutputResultFileName>")
        print('Example: topsis data.xlsx "1,1,1,2" "+,+,-,+" result.xlsx')
        sys.exit(1)

    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    output_file = sys.argv[4]

    try:
        topsis(input_file, weights, impacts, output_file)
        print(f"Result saved in: {output_file}")
    except TopsisError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
