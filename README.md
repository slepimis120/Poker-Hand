# Poker Hands Analysis

This repository contains code, datasets, and Jupyter notebooks for an analysis project focused on bluff detection in Texas Hold'em poker.

## Folder Structure

- **data/**
  Contains the input CSV files with poker hands data:
  - `poker_dataset.csv` (hand strength classification)
  - `poker_hand_history_dataset.csv` (historical hand logs with detailed actions)

- **src/**
  Python scripts for data analysis and preprocessing:
  - `poker_eda.py` – EDA on poker_dataset.csv
  - `poker_hand_history_eda.py` – EDA on poker_hand_history_dataset.csv

- **notebooks/**
  Jupyter Notebook files with complete analyses, visualizations, and experiments:
  - `poker_eda.ipynb`
  - `poker_hand_history_eda.ipynb`

## Project Overview

- **Goal:**
  Explore contextual features and patterns in No-Limit Texas Hold'em hands and analyze the possibility of bluff actions. Build a supervised model for bluff detection based on statistics and behavior extracted from hand history logs.
- **Methods:**
  Includes descriptive analysis, visualization, and baseline machine learning models (such as RandomForest) to identify features associated with bluffing: aggression, number of players, stacks, showdown frequency, etc.

## Usage

1. Make sure you have the required Python libraries:
    ```
    pip install pandas matplotlib seaborn
    ```

2. Run scripts from the `src/` folder or open notebooks in the `notebooks/` folder.

3. Adjust file paths to the `data/` directory as needed.