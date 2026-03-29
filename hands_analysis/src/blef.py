from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd

rank_map = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "t": 10, "j": 11, "q": 12, "k": 13, "a": 14
}

hand_order = {
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

def parse_card(card):
    card = str(card).strip().lower()
    if len(card) != 2:
        return None
    return (rank_map[card[0]], card[1])

def evaluate_5(cards):
    parsed = [parse_card(c) for c in cards]
    if any(c is None for c in parsed):
        return None

    ranks = [r for r, s in parsed]
    suits = [s for r, s in parsed]
    counts = Counter(ranks)
    vals = sorted(counts.values(), reverse=True)

    is_flush = len(set(suits)) == 1

    unique_ranks = sorted(set(ranks))
    is_straight = False
    high_straight = None

    if len(unique_ranks) == 5:
        if unique_ranks[-1] - unique_ranks[0] == 4:
            is_straight = True
            high_straight = unique_ranks[-1]
        elif unique_ranks == [2, 3, 4, 5, 14]:
            is_straight = True
            high_straight = 5

    if is_straight and is_flush:
        if high_straight == 14:
            return ("ROYAL FLUSH", 14)
        return ("STRAIGHT FLUSH", high_straight)

    if vals == [4, 1]:
        four = max(r for r, c in counts.items() if c == 4)
        return ("FOUR OF A KIND", four)

    if vals == [3, 2]:
        three = max(r for r, c in counts.items() if c == 3)
        return ("FULL HOUSE", three)

    if is_flush:
        return ("FLUSH", max(ranks))

    if is_straight:
        return ("STRAIGHT", high_straight)

    if vals == [3, 1, 1]:
        three = max(r for r, c in counts.items() if c == 3)
        return ("THREE OF A KIND", three)

    if vals == [2, 2, 1]:
        pairs = sorted([r for r, c in counts.items() if c == 2], reverse=True)
        return ("TWO PAIR", pairs[0])

    if vals == [2, 1, 1, 1]:
        pair = max(r for r, c in counts.items() if c == 2)
        return ("PAIR", pair)

    return ("NOTHING", max(ranks))

def best_hand_class(cards):
    if len(cards) < 5:
        return np.nan

    best = None

    for combo in combinations(cards, 5):
        result = evaluate_5(combo)
        if result is None:
            continue
        label, tiebreak = result
        score = (hand_order[label], tiebreak)
        if best is None or score > best:
            best = score

    if best is None:
        return np.nan

    label, tiebreak = None, None
    for combo in combinations(cards, 5):
        result = evaluate_5(combo)
        if result is None:
            continue
        cur_label, cur_tiebreak = result
        if (hand_order[cur_label], cur_tiebreak) == best:
            label, tiebreak = cur_label, cur_tiebreak
            break

    if label == "PAIR":
        return "LOW_PAIR" if tiebreak <= 9 else "HIGH_PAIR"

    return label

def max_card_from_hole(cards):
    return max(rank_map[c[0].lower()] for c in cards)

def same_suit_from_hole(cards):
    return 1 if cards[0][1].lower() == cards[1][1].lower() else 0

action_cols = ["action_pre", "action_flop", "action_turn", "action_river"]

def is_aggressive_action(row):
    for col in action_cols:
        val = str(row[col]).lower()
        if ("bet" in val) or ("raise" in val):
            return 1
    return 0