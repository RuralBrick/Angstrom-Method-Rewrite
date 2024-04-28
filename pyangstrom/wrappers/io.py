from pathlib import Path
import csv


def load_exp_condition(p_exp_cond: Path) -> list[dict]:
    with open(p_exp_cond, newline='') as f:
        reader = csv.reader(f)
        keys = next(reader)
        exp_cond = []
        for row in reader:
            record = {key: value for key, value in zip(keys, row)}
            exp_cond.append(record)
    return exp_cond
