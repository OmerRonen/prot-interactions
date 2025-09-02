# Protein Interactions Project

This project uses ESM (Evolutionary Scale Modeling) for protein sequence analysis and structure prediction.

## Environment Setup

This project uses [Pixi](https://pixi.sh/) for dependency management and environment setup.

### Prerequisites

- Pixi (install with: `curl -fsSL https://pixi.sh/install.sh | bash`)

### Installation

1. Clone or navigate to the project directory
2. Install dependencies:
   ```bash
   pixi install
   ```

### TODOs

The only function that needs to be implemented is that `get_feature_importance` which takes token and the preiction index, which is the residue position on the protein. 
This function should return a tensor with the same length as the tokens.