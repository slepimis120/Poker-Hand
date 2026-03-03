import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# ---------- Config ----------
CSV_PATH = "../data/poker_hand_history_dataset.csv"

# ---------- Helper functions ----------
def str_to_list(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return []

def parse_actions(actions):
    if not isinstance(actions, list):
        try:
            actions = ast.literal_eval(actions)
        except Exception:
            return {"bet":0, "raise":0, "fold":0, "showdown":0, "aggressive":0}
    bet, raise_, fold, showdown, aggressive = 0, 0, 0, 0, 0
    for act in actions:
        act_l = act.lower()
        if "cbr" in act_l or "bet" in act_l:
            bet += 1
            aggressive += 1
        if "raise" in act_l:
            raise_ += 1
            aggressive += 1
        if "f" in act_l and "fold" in act_l or act_l.endswith(" f"):
            fold += 1
        if "sm" in act_l:
            showdown += 1
    return {"bet":bet, "raise":raise_, "fold":fold, "showdown":showdown, "aggressive":aggressive}

def basic_info(df):
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Number of hands:", len(df))
    print("Venues:", df['venue'].value_counts().to_dict())
    print("Sample row:\n", df.iloc[0])

def plot_distribution(df, column, top_n=10):
    counts = df[column].value_counts().head(top_n)
    plt.figure(figsize=(8,4))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.title(f"Top {top_n} {column} Values")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# ---------- Analysis ----------
if __name__ == "__main__":
    # Load CSV with stringified lists
    converters = {
        "blinds_or_straddles": str_to_list,
        "starting_stacks": str_to_list,
        "seats": str_to_list,
        "players": str_to_list,
        "actions": str_to_list,
        "winnings": str_to_list,
    }
    df = pd.read_csv(CSV_PATH, converters=converters)
    basic_info(df)

    # Add number of players, average stack/big blind
    df["num_players"] = df["players"].apply(len)
    df["avg_stack"] = df["starting_stacks"].apply(lambda x: sum(x)/len(x) if isinstance(x, list) and len(x)>0 else 0)
    df["big_blind"] = df["blinds_or_straddles"].apply(lambda x: max(x) if isinstance(x, list) and len(x)>0 else 0)
    df["pot_size"] = df["winnings"].apply(lambda x: sum(x) if isinstance(x, list) else 0)

    # Analysis of number of players, average stack, and big blind
    print('\nPlayer count distribution:', df["num_players"].value_counts())
    print("Big blind min/max/mean:", df["big_blind"].min(), df["big_blind"].max(), df["big_blind"].mean())
    print("Average stack min/max/mean:", df["avg_stack"].min(), df["avg_stack"].max(), df["avg_stack"].mean())

    plot_distribution(df, "num_players")

    # Parse action stats
    rekap = df["actions"].apply(parse_actions).apply(pd.Series)
    for col in ["bet","raise","fold","showdown","aggressive"]:
        df[col] = rekap[col]

    print("\nAverage number of bets per hand: ", df['bet'].mean())
    print("Average number of raises per hand: ", df['raise'].mean())
    print("Average number of folds per hand: ", df['fold'].mean())
    print("Percentage of hands with showdown: ", (df['showdown']>0).mean()*100, "%")

    plt.figure(figsize=(6,4))
    sns.histplot(df['aggressive'], bins=range(0,df['aggressive'].max()+2), kde=False)
    plt.title("Distribution of aggressive actions per hand")
    plt.xlabel("Aggressive actions (bet/raise) per hand")
    plt.ylabel("Number of hands")
    plt.tight_layout()
    plt.show()

    # Approximate "bluff" label: hands with at least one aggressive action but no showdown
    df["possible_bluff"] = ((df['showdown']==0) & (df['aggressive']>0)).astype(int)
    print("Percentage of 'possible bluff' hands: ", df['possible_bluff'].mean()*100, "%")

    plt.figure(figsize=(4,4))
    sns.barplot(x=["Non-bluff","Possible bluff"], y=df["possible_bluff"].value_counts(normalize=True).values*100)
    plt.ylabel("% of hands")
    plt.title("Approximate bluff hands (aggression without showdown)")
    plt.show()

    # Additional features can be engineered for ML from these columns

    # Save for further use (optional)
    df.to_csv("../data/phh_analysis_output.csv", index=False)