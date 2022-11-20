from unittest.mock import DEFAULT
import matplotlib.pyplot as plt

# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--clf_name',type=str, required=True)
parser.add_argument('--Random_state', type=int, required=True)
# Parse the argument
args = parser.parse_args()
print("clf_name::",args.clf_name)
print("Random_state::", args.Random_state)

