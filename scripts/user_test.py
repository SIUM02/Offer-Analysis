import argparse
import collections
import csv
import math
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOGUE = os.path.join(BASE, "data", "all_operators_official.csv")


USABLE_CATEGORIES = {"Data", "Combo", "Minute", "SMS", "Unlimited"}


UNLIMITED_GB = 1000.0


RESTRICTED_PATTERN = (
    r"\bonly\b|youtube|tiktok|\bimo\b|\bbip\b|telegram|facebook|messenger"
    r"|hoichoi|bongo|deeptoplay|iscreen|toffee|binge|chorki|lionsgate"
    r"|sonyliv|t-sports|subscription|streaming|content|play pack"
)


MISS_PENALTY = 1e6
SHORTFALL_PENALTY = 5.0
WASTE_PENALTY = 0.15
VALIDITY_PENALTY = 0.40
SMS_TOPUP_BDT = 1.00

OPERATOR_ALIASES = {
    "1": "Grameenphone", "gp": "Grameenphone", "grameenphone": "Grameenphone",
    "2": "Banglalink", "bl": "Banglalink", "banglalink": "Banglalink",
    "3": "Teletalk", "tt": "Teletalk", "teletalk": "Teletalk",
    "4": "Robi", "robi": "Robi",
}

QUIT = {"q", "quit", "exit"}


class Quit(Exception):
    pass


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_catalogue():
    if not os.path.exists(CATALOGUE):
        raise SystemExit(f"Catalogue not found: {CATALOGUE}")

    restricted = re.compile(RESTRICTED_PATTERN, re.IGNORECASE)

    offers = []
    with open(CATALOGUE, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["category"] not in USABLE_CATEGORIES:
                continue
            if restricted.search(raw["offer_name"] or ""):
                continue

            offer = dict(raw)
            for column in ("data_gb", "minutes", "sms", "validity_days",
                           "price_bdt", "price_per_gb", "price_per_minute",
                           "price_per_sms"):
                offer[column] = number(raw.get(column))

            if offer["validity_days"] <= 0:
                offer["validity_days"] = 30.0

            offer["effective_gb"] = (UNLIMITED_GB
                                     if offer["category"] == "Unlimited"
                                     else offer["data_gb"])
            offers.append(offer)

    if not offers:
        raise SystemExit(f"No buyable packs in {CATALOGUE}")
    return offers


def quartile(values):
    ordered = sorted(values)
    if not ordered:
        return None
    position = 0.25 * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def reference_rates(catalogue):
    def rate(offers, quantity, price, others, minimum):
        pure = [o[price] for o in offers
                if o[quantity] >= minimum
                and o[others[0]] == 0 and o[others[1]] == 0]
        return quartile(pure)

    fallbacks = (
        rate(catalogue, "data_gb", "price_per_gb", ("minutes", "sms"), 1) or 30.0,
        rate(catalogue, "minutes", "price_per_minute", ("data_gb", "sms"), 30) or 1.0,
        rate(catalogue, "sms", "price_per_sms", ("data_gb", "minutes"), 50) or 0.3,
    )

    rates = {}
    for operator in {o["operator"] for o in catalogue}:
        group = [o for o in catalogue if o["operator"] == operator]
        rates[operator] = (
            rate(group, "data_gb", "price_per_gb", ("minutes", "sms"), 1) or fallbacks[0],
            rate(group, "minutes", "price_per_minute", ("data_gb", "sms"), 30) or fallbacks[1],
            rate(group, "sms", "price_per_sms", ("data_gb", "minutes"), 50) or fallbacks[2],
        )
    return rates


def score_offer(offer, wants, rates, hard=True):
    data_want, minute_want, sms_want, validity = wants
    gb_rate, min_rate, sms_rate = rates
    validity = max(validity, 1)

    repeats = repeats_needed(offer, validity)

    price = offer["price_bdt"] * repeats
    gb = offer["effective_gb"] * repeats
    minutes = offer["minutes"] * repeats
    sms = offer["sms"] * repeats

    short_gb = max(data_want - gb, 0.0)
    short_min = max(minute_want - minutes, 0.0)
    short_sms = max(sms_want - sms, 0.0)

    shortfall = SHORTFALL_PENALTY * (
        short_gb * gb_rate + short_min * min_rate + short_sms * sms_rate)
    if hard and (short_gb > 0 or short_min > 0 or short_sms > 0):
        shortfall += MISS_PENALTY

    waste = WASTE_PENALTY * min(
        max(gb - data_want, 0.0) * gb_rate
        + max(minutes - minute_want, 0.0) * min_rate
        + max(sms - sms_want, 0.0) * sms_rate,
        price,
    )

    excess = max(offer["validity_days"] / validity - 1, 0.0)
    validity_cost = VALIDITY_PENALTY * math.log1p(excess) * price

    return price + shortfall + waste + validity_cost


def repeats_needed(offer, validity_want):
    return max(1, math.ceil(max(validity_want, 1) / offer["validity_days"]))


def shortfall(offer, wants):
    data_want, minute_want, sms_want, validity_want = wants
    repeats = repeats_needed(offer, validity_want)
    return (max(data_want - offer["effective_gb"] * repeats, 0.0),
            max(minute_want - offer["minutes"] * repeats, 0.0),
            max(sms_want - offer["sms"] * repeats, 0.0))


def covers(offer, wants):
    return not any(shortfall(offer, wants))


def period_price(offer, validity_want):
    return offer["price_bdt"] * repeats_needed(offer, validity_want)


def cheapest_filling(packs, column, amount, validity_want):
    affordable = [(period_price(o, validity_want), o) for o in packs
                  if o[column] * repeats_needed(o, validity_want) >= amount]
    if not affordable:
        return None
    price, pack = min(affordable, key=lambda pair: pair[0])
    return pack, price


TopUp = collections.namedtuple("TopUp", "what price how pack_id")


def topup(offer, wants, packs, rates):
    gb_rate, minute_rate, sms_rate = rates
    sms_rate = max(sms_rate, SMS_TOPUP_BDT)
    validity_want = wants[3]

    lines = []
    for amount, column, unit, rate in zip(
            shortfall(offer, wants),
            ("effective_gb", "minutes", "sms"),
            ("GB", "min", "SMS"),
            (gb_rate, minute_rate, sms_rate)):
        if not amount:
            continue
        what = f"{amount:g} {unit}"
        filler = cheapest_filling(packs, column, amount, validity_want)
        if filler:
            pack, price = filler
            lines.append(TopUp(what, price, pack["offer_name"], pack["offer_id"]))
        else:
            lines.append(TopUp(what, amount * rate,
                               f"pay as you go at BDT {rate:g}", None))
    return lines


def plan(offer, wants, packs, rates):
    return frozenset({offer["offer_id"]}
                     | {line.pack_id
                        for line in topup(offer, wants, packs, rates)
                        if line.pack_id})


def allin_cost(offer, wants, packs, rates):
    return (period_price(offer, wants[3])
            + sum(line.price for line in topup(offer, wants, packs, rates)))


def recommend(catalogue, rates, operator, wants, top=3, strict=False):
    sellable = [o for o in catalogue if o["operator"] == operator]
    if not sellable:
        return []

    rate = rates[operator]
    if strict:
        ranked = sorted(sellable,
                        key=lambda o: score_offer(o, wants, rate))[:top]
    else:
        ordered = sorted(sellable,
                         key=lambda o: (allin_cost(o, wants, sellable, rate),
                                        o["price_bdt"]))
        ranked, seen = [], set()
        for offer in ordered:
            key = plan(offer, wants, sellable, rate)
            if key in seen:
                continue
            seen.add(key)
            ranked.append(offer)
            if len(ranked) == top:
                break

    costs = [allin_cost(o, wants, sellable, rate) for o in ranked]
    cheapest = min(costs)
    return [(offer, min(cheapest / max(cost, 1e-9), 1.0))
            for offer, cost in zip(ranked, costs)]


def show(row, value, wants, packs, rates, heading, nothing_covers=False):
    data_want, minute_want, sms_want, validity_want = wants
    unlimited = row["category"] == "Unlimited"
    data = "Unlimited" if unlimited else f"{row['data_gb']:g} GB"

    print(f"\n{heading}")
    print(f"  {row['offer_name']}")
    print(f"    Operator   : {row['operator']}")
    print(f"    Data       : {data}")
    print(f"    Minutes    : {row['minutes']:g}")
    print(f"    SMS        : {row['sms']:g}")
    print(f"    Validity   : {row['validity_days']:g} days")
    print(f"    Price      : BDT {row['price_bdt']:g}")
    print(f"    USSD       : {row['ussd_code']}")
    print(f"    Value      : {value:.0%}")

    repeats = repeats_needed(row, validity_want)
    if repeats > 1:
        total_data = "Unlimited" if unlimited else f"{row['data_gb'] * repeats:g} GB"
        print(f"    To cover {validity_want:g} days: buy {repeats}x -> "
              f"{total_data}, {row['minutes'] * repeats:g} min, "
              f"{row['sms'] * repeats:g} SMS, "
              f"BDT {row['price_bdt'] * repeats:g} total")

    missing = topup(row, wants, packs, rates)
    if missing:
        label = "    Top up     : "
        for line in missing:
            how = line.how if line.pack_id is None else f'add "{line.how}"'
            print(f"{label}{line.what} short -> {how}, BDT {line.price:g}")
            label = " " * len(label)
        allin = period_price(row, validity_want) + sum(l.price for l in missing)
        note = (f"  ({row['operator']} sells nothing that covers this)"
                if nothing_covers else "")
        print(f"    All in     : BDT {allin:g}{note}")


def report(catalogue, rates, operator, wants, strict=False):
    data_gb, minutes, sms, validity = wants
    print(f"\nRequest: {operator} | {data_gb:g} GB | {minutes:g} min | "
          f"{sms:g} SMS | {validity:g} days")

    results = recommend(catalogue, rates, operator, wants, strict=strict)
    if not results:
        print(f"  {operator} sells no buyable pack in this catalogue.")
        return

    packs = [o for o in catalogue if o["operator"] == operator]
    nothing_covers = not any(covers(o, wants) for o in packs)

    for rank, (row, value) in enumerate(results):
        show(row, value, wants, packs, rates[operator],
             "RECOMMENDED" if rank == 0 else f"Alternative {rank}",
             nothing_covers=nothing_covers)


def ask(prompt, default=0.0):
    while True:
        raw = input(prompt).strip()
        if raw.lower() in QUIT:
            raise Quit
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("   Please enter a number (or 'q' to quit).")
            continue
        if value < 0:
            print("   Please enter zero or more.")
            continue
        return value


def ask_operator(known):
    while True:
        raw = input("  Operator [1 Grameenphone / 2 Banglalink / "
                    "3 Teletalk / 4 Robi]: ").strip()
        if raw.lower() in QUIT:
            raise Quit
        operator = OPERATOR_ALIASES.get(raw.lower())
        if operator in known:
            return operator
        print("   Please choose 1-4, or type the operator's name.")


def parse_args(known):
    parser = argparse.ArgumentParser(
        description="Match a request against the pack catalogue.")
    parser.add_argument("--operator", help="Grameenphone, Banglalink, "
                                           "Teletalk or Robi (or 1-4)")
    parser.add_argument("--data", type=float, default=0.0, help="GB wanted")
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--sms", type=float, default=0.0)
    parser.add_argument("--validity", type=float, default=30.0,
                        help="days to cover (default 30)")
    parser.add_argument("--strict", action="store_true",
                        help="offer only packs that meet the request in "
                             "full, ignoring what a top-up would cost")
    args = parser.parse_args()

    if args.operator is not None:
        operator = OPERATOR_ALIASES.get(args.operator.strip().lower())
        if operator not in known:
            parser.error(f"unknown operator {args.operator!r}; "
                         f"choose one of {', '.join(known)}")
        args.operator = operator
    return args


def main():
    catalogue = load_catalogue()
    rates = reference_rates(catalogue)
    known = sorted({o["operator"] for o in catalogue})
    args = parse_args(known)

    print("\nSIM offer recommender")
    print(f"Matching against {len(catalogue)} buyable packs "
          f"({', '.join(known)}). No model -- matching algorithm only.")

    if args.operator:
        report(catalogue, rates, args.operator,
               (args.data, args.minutes, args.sms, args.validity),
               strict=args.strict)
        return

    print("Enter what you want. Blank = 0, 'q' = quit.\n")
    asked = 0
    while True:
        try:
            operator = ask_operator(known)
            data_gb = ask("  Data wanted (GB)     : ")
            minutes = ask("  Minutes wanted       : ")
            sms = ask("  SMS wanted           : ")
            validity = ask("  Validity (days) [30] : ", default=30.0)
        except EOFError:
            if asked == 0:
                raise SystemExit(
                    "\nNothing to read from -- this run has no keyboard.\n"
                    "Pass the request on the command line instead:\n"
                    "  python3 scripts/user_test.py --operator Teletalk "
                    "--data 5 --minutes 250 --sms 50 --validity 30")
            print("\nBye.")
            return
        except (Quit, KeyboardInterrupt):
            print("\nBye.")
            return

        asked += 1

        report(catalogue, rates, operator, (data_gb, minutes, sms, validity),
               strict=args.strict)
        print("\n" + "-" * 68)


if __name__ == "__main__":
    main()
