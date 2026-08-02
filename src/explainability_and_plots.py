import os
import sys

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate_paper_graphs import generate_all_paper_graphs

if __name__ == '__main__':
    print("Executing comprehensive paper-aligned graph generation...")
    generate_all_paper_graphs()
