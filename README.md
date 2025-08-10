# Fantacalcio Simulation Suite

## Overview

Fantacalcio Simulation Suite is a powerful command-line tool for fantasy football ("Fantacalcio") auction preparation. It has evolved from a simple web scraper into a sophisticated analysis and simulation engine designed to give you a competitive edge in your auction.

This tool does **not** scrape data. Instead, it uses a local Excel file of player data to run a series of complex analyses, providing deep insights into player valuation, auction dynamics, and optimal bidding strategies.

## Key Features

- **Configurable Simulation**: All aspects of the simulation are controlled through a detailed `config.yaml` file, allowing you to tailor the analysis to your league's specific rules (e.g., number of teams, budget, formations).
- **Advanced Analysis Modules**:
  - **Tier System**: Automatically classifies players into tiers (Elite, High, Medium, Low) based on their stats.
  - **Scarcity Analysis**: Calculates player value based on the scarcity of their role in the league.
  - **Shading Simulation**: Uses Monte Carlo simulation to model opponent bidding behavior and recommend optimal bid "shading" (how much to bid over a player's base value).
  - **Auction Flow Analysis**: Simulates the overall flow of the auction to help with budget management.
  - **Auction Strategy**: Helps identify the best players to target based on your chosen formation.
- **Bidding Recommendations**:
  - Generates a prioritized list of players to bid on for the first round of your auction.
  - Includes a "smart bidding" feature that automatically adjusts bids to avoid round numbers (e.g., a bid of 50 will be adjusted to 51), a common strategy to win auctions.
- **Comprehensive Reporting**: Generates a detailed PDF report summarizing the results of all analyses, complete with visualizations.
- **Auditable Results**: Creates an audit trail to ensure results are reproducible, tracking the source data hash and git commit.

## How to Use

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Your Data**:
    - Obtain an Excel file with player listings and stats for the upcoming season.
    - Update the `excel_path` and `data_path` in `config.yaml` to point to your file. The tool expects the data to start on the second row (header=1).

3.  **Configure Your League**:
    - Edit `config.yaml` to match your league's settings (e.g., `num_teams`, `total_budget`, `primary_module` for your formation).

4.  **Run the Simulation**:
    - The main entry point is `sim.py`. You must provide a path to your configuration file and specify which analyses to run.
    - To run all analyses and generate a full report, use the `--all` flag:
      ```bash
      python sim.py --config config.yaml --all
      ```
    - You can also run specific analyses:
      ```bash
      python sim.py --config config.yaml --scarcity --shading
      ```
    - The results, including the PDF report and CSV files, will be saved in the `reports/` directory by default.

## Legacy Scraper (`Fantacalcio.py`)

This repository also contains the original web-scraping script, `Fantacalcio.py`. This script scrapes data from `fantacalciopedia.com` and calculates a simple "convenience" metric.

**Note**: This script is considered **deprecated** and is no longer maintained. The main focus of this project is the `sim.py` simulation suite, which provides far more powerful and reliable analysis.

---
*Nato da un'idea di cttynul*
