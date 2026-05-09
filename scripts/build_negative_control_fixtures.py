"""Build the two negative-control fixtures used in Appendix D.

Both are 30-question JSONL files in the GSM8K-compatible format
({"question": ..., "answer": "#### <gold>"}). They're intended as
calibration tasks for HCDS — on a task where hidden multi-step
reasoning is implausible, HCDS should be near zero. If it isn't, that
tells us HCDS is partly measuring stylistic length / entropy
differences rather than reasoning.

Datasets:
  arithmetic_30.jsonl    — single-step arithmetic. Gold is exact integer.
  factual_30.jsonl       — well-known facts with numeric answers. Gold
                            is the integer mentioned in the standard
                            answer (year, count, etc.).
"""

import json
import random
from pathlib import Path

OUT = Path("data/fixtures/negative_control")
OUT.mkdir(parents=True, exist_ok=True)


def write_jsonl(name, rows):
    path = OUT / name
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


# ----------------------------- arithmetic ------------------------------

rng = random.Random(17)
arith = []
ops = [("+", lambda a, b: a + b),
       ("-", lambda a, b: a - b),
       ("\\times", lambda a, b: a * b)]

while len(arith) < 50:
    op_str, op_fn = rng.choice(ops)
    if op_str == "\\times":
        a, b = rng.randint(2, 12), rng.randint(2, 12)
    elif op_str == "-":
        a = rng.randint(20, 99)
        b = rng.randint(2, a - 1)
    else:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
    gold = op_fn(a, b)
    q = f"What is ${a} {op_str} {b}$?"
    ans = f"#### {gold}"
    if (q, ans) not in [(r["question"], r["answer"]) for r in arith]:
        arith.append({"question": q, "answer": ans})

write_jsonl("arithmetic_50.jsonl", arith)


# -------------------------- factual lookups ---------------------------
# Numeric-answer facts. Each gold is verifiable. We avoid anything
# that requires multi-step reasoning (so no "how old was X when Y").

factual = [
    ("How many states are in the United States of America?", 50),
    ("In what year did humans first land on the Moon?", 1969),
    ("How many planets are in our solar system?", 8),
    ("How many continents are there?", 7),
    ("How many bones are there in the adult human body?", 206),
    ("How many time zones are there in the world?", 24),
    ("How many sides does a hexagon have?", 6),
    ("How many degrees are in a right angle?", 90),
    ("How many wonders are there in the ancient world?", 7),
    ("In what year did World War II end?", 1945),
    ("In what year was the United States founded (declaration of independence)?", 1776),
    ("How many keys are on a standard piano?", 88),
    ("How many chambers does the human heart have?", 4),
    ("How many strings does a standard guitar have?", 6),
    ("How many teeth does an adult human have?", 32),
    ("How many minutes are in an hour?", 60),
    ("How many seconds are in a minute?", 60),
    ("How many days are in a non-leap year?", 365),
    ("How many days are in February in a leap year?", 29),
    ("How many feet are in a yard?", 3),
    ("How many inches are in a foot?", 12),
    ("How many ounces are in a pound?", 16),
    ("How many sides does a triangle have?", 3),
    ("How many sides does an octagon have?", 8),
    ("How many degrees are in a full circle?", 360),
    ("How many letters are in the English alphabet?", 26),
    ("How many strings does a violin have?", 4),
    ("How many players are on a soccer team on the field per side?", 11),
    ("How many quarts are in a gallon?", 4),
    ("In what year did the Berlin Wall fall?", 1989),
    ("How many holes are on a standard golf course?", 18),
    ("How many sides does a pentagon have?", 5),
    ("How many primary colors are there in the additive RGB model?", 3),
    ("How many vertebrae are in the human spinal column?", 33),
    ("How many millimeters are in a centimeter?", 10),
    ("How many liters are in a kiloliter?", 1000),
    ("In what year did the Titanic sink?", 1912),
    ("In what year did the first iPhone launch?", 2007),
    ("How many books are in the Lord of the Rings trilogy?", 3),
    ("How many movies are in the original Star Wars trilogy?", 3),
    ("How many symphonies did Beethoven compose?", 9),
    ("How many players are on a basketball team on the court per side?", 5),
    ("How many sides does a square have?", 4),
    ("How many years are in a decade?", 10),
    ("How many years are in a century?", 100),
    ("How many years are in a millennium?", 1000),
    ("How many ribs does an adult human typically have?", 24),
    ("How many cards are in a standard deck (excluding jokers)?", 52),
    ("In what year did the United States declare independence?", 1776),
    ("How many countries are members of the United Nations as of 2024?", 193),
]

assert len(factual) == 50, len(factual)
factual_rows = [{"question": q, "answer": f"#### {g}"} for q, g in factual]
write_jsonl("factual_50.jsonl", factual_rows)
