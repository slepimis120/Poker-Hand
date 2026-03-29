import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def basic_info(df):
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    # print("\nMissing values:\n", df.isnull().sum())

def plot_distribution(df, column):
    counts = df[column].value_counts()
    percentages = df[column].value_counts(normalize=True) * 100

    print(f"\nDistribution for {column}:")
    print(pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    }))

    game_phase = {
        "result1": "Flop",
        "result2": "Turn",
        "result3": "River",
        "result1_class": "Flop",
        "result2_class": "Turn",
        "result3_class": "River"
    }

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(f"Distribution of {game_phase[column]} Hand Strength")
    plt.xticks(rotation=45)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

card_value = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '10': 10,
    'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

# Razdvajanje PAIR na LOW_PAIR i HIGH_PAIR
def classify_pair(row, col):
    if row[col] != "PAIR":
        return row[col]
    cards = row['hand'].split()
    ranks = [card[1:] for card in cards]
    pair_value = max([card_value[r] for r in ranks])
    return "LOW_PAIR" if pair_value <= 9 else "HIGH_PAIR"

hand_rank = {
    "NOTHING": 0,
    "PAIR": 1,
    "TWO PAIR": 2,
    "THREE OF A KIND": 3,
    "STRAIGHT": 4,
    "FLUSH": 5,
    "FULL HOUSE": 6,
    "FOUR OF A KIND": 7,
    "STRAIGHT FLUSH": 8,
    "ROYAL FLUSH": 9
}

def strength_category(hand_class):
    if hand_class in ["NOTHING", "LOW_PAIR"]:
        return "Weak"
    elif hand_class in ["HIGH_PAIR", "TWO PAIR"]:
        return "Strong"
    else:
        return "Premium"

if __name__ == "__main__":

    data_file = Path(__file__).resolve().parent.parent / "data" / "poker_dataset.csv"
    df = load_data(data_file)
    print("Initial Data:")
    basic_info(df)
    # plot_distribution(df, "result1")
    # plot_distribution(df, "result2")
    # plot_distribution(df, "result3")

    df["rank1"] = df["result1"].map(hand_rank)
    df["rank2"] = df["result2"].map(hand_rank)
    df["rank3"] = df["result3"].map(hand_rank)

    # FLOP -> TURN
    improve_12 = (df["rank2"] > df["rank1"]).sum()
    same_12 = (df["rank2"] == df["rank1"]).sum()
    worse_12 = (df["rank2"] < df["rank1"]).sum()

    print("\nFlop → Turn")
    print("Improved:", improve_12)
    print("Same:", same_12)
    print("Worse:", worse_12)

    # TURN -> RIVER
    improve_23 = (df["rank3"] > df["rank2"]).sum()
    same_23 = (df["rank3"] == df["rank2"]).sum()
    worse_23 = (df["rank3"] < df["rank2"]).sum()

    print("\nTurn → River")
    print("Improved:", improve_23)
    print("Same:", same_23)
    print("Worse:", worse_23)

    worse_23_rows = df[df["rank3"] < df["rank2"]]

    worse_13_rows = df[df["rank3"] < df["rank1"]]

    # BRISANJE REDOVA GDE JE FLOP BOLJI OD TURN-a (rank2 < rank1)
    df_clean = df[df["rank2"] >= df["rank1"]]

    # BRISANJE REDOVA GDE JE TURN BOLJI OD RIVER-a (rank3 < rank2)
    df_clean = df_clean[df_clean["rank3"] >= df_clean["rank2"]]

    # BRISANJE REDOVA GDE JE FLOP BOLJI OD RIVER-a (rank3 < rank1) - ovi slucajevi su verovatno vec uklonejni gore, ali ovo je dodatna provera
    df_clean = df_clean[df_clean["rank3"] >= df_clean["rank1"]]

    print("Novi broj redova:", df_clean.shape[0])

    for col in ['result1', 'result2', 'result3']:
        df_clean[f'{col}_class'] = df_clean.apply(lambda row: classify_pair(row, col), axis=1)

    # plot_distribution(df_clean, "result1_class")
    # plot_distribution(df_clean, "result2_class")
    # plot_distribution(df_clean, "result3_class")

    # PRIKAZ PIE CHARTA NA RIVERU UZ KLASIFIKOVAN HAND_STRENGTH
    df_clean["river_strength"] = df_clean["result3_class"].apply(strength_category)
    strength_counts = df_clean["river_strength"].value_counts()
    strength_percent = df_clean["river_strength"].value_counts(normalize=True) * 100

    print(pd.DataFrame({
        "Count": strength_counts,
        "Percentage": strength_percent.round(2)
    }))

    plt.figure(figsize=(8,8))
    plt.pie(
        strength_counts,
        labels=strength_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("River Hand Strength Distribution")
    plt.axis("equal")
    plt.show()

    # PRIKAZIVANJE MATRICE TRANZICIJE FLOP → RIVER
    # Mapiranje rank nazad u string
    rank_to_hand = {v: k for k, v in hand_rank.items()}

    # Kreiranje transition matrice sa rankovima
    transition_matrix_pct = pd.crosstab(
        df_clean["rank1"],  # result1
        df_clean["rank3"],  # result3
        normalize="index"
    ) * 100

    # Zamena redova i kolona stringovima
    transition_matrix_pct.index = transition_matrix_pct.index.map(rank_to_hand)
    transition_matrix_pct.columns = transition_matrix_pct.columns.map(rank_to_hand)

    # Prikaz heatmap-a
    plt.figure(figsize=(10,8))
    sns.heatmap(transition_matrix_pct.round(1), annot=True, fmt=".1f", cmap="Blues")
    plt.title("Flop → River Transition Matrix (%)")
    plt.ylabel("Flop")
    plt.xlabel("River")
    # plt.show()

    df_clean = df_clean.copy()

    df_clean["max_card"] = df_clean["hand"].apply(
        lambda x: max(card_value[card[1:]] for card in x.split())
    )

    df_clean["same_suit"] = df_clean["hand"].apply(
        lambda x: 1 if len(set(card[0] for card in x.split())) == 1 else 0
    )

    y = df_clean["river_strength"]  # Weak / Strong / Premium

    X_cat = pd.get_dummies(df_clean[["result1_class", "result2_class"]])
    X_num = df_clean[["max_card", "same_suit"]]

    X = pd.concat([X_cat, X_num], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
