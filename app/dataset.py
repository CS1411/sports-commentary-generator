"""Milestone 1: simulated sports events paired with commentary.

Every example is built from a template, so the data is synthetic, reproducible and free of real people.
Each commentary is guaranteed to mention the key facts of its event (names, teams, score), which is what
the evaluation later checks for.
"""
import json
import random
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "commentary.jsonl"

FIRST = ["Alex", "Bruno", "Carlos", "Dmitri", "Emil", "Felix", "Gabriel", "Hugo", "Ivan", "Jonas",
         "Kai", "Luca", "Mateo", "Nico", "Oscar", "Pavel"]
LAST = ["Alvarez", "Brandt", "Castillo", "Dubois", "Eriksen", "Fontaine", "Gallo", "Hartmann", "Ibarra",
        "Jansen", "Kovac", "Larsen", "Moreau", "Novak", "Okafor", "Petrov"]
TEAMS = {
    "football": ["Riverside FC", "Harbor United", "Northgate Rovers", "Kingsford City", "Redhill Athletic",
                 "Oakfield Town"],
    "basketball": ["Metro Hawks", "Bay Wolves", "Summit Kings", "Coastal Sharks", "Iron Bulls", "Pine Owls"],
    "cricket": ["Highland Lions", "Valley Royals", "Delta Stars", "Harbor Hawks", "Desert Eagles", "Lake Tigers"],
}

# type -> (sport, event template, facts, {tone: [commentary templates]})
# Fields: team, opp, player, other, clock, a, b.  "score" is "{a}-{b}".
EVENT_TYPES = {
    "goal": ("football", "{clock} GOAL! {player} scores for {team} against {opp}. It is now {score}.",
             ["player", "team", "score"], {
                 "excited": ["{player} scores! What a strike for {team}, and it is {score} against {opp}!",
                             "Goal for {team}! {player} finds the net and the score is {score} against {opp}!"],
                 "calm": ["{player} scores for {team}. The score is {score} against {opp}.",
                          "A composed finish from {player}, and {team} make it {score} against {opp}."]}),
    "penalty_save": ("football", "{clock} SAVE: {player} of {team} saves a penalty from {other} of {opp}.",
                     ["player", "other", "team"], {
                         "excited": ["{player} saves it! {other} cannot beat the keeper from the spot and "
                                     "{team} are still in this!",
                                     "What a stop by {player}! {other} is denied from the penalty spot and "
                                     "{team} survive!"],
                         "calm": ["{player} saves the penalty from {other}. {team} survive the scare.",
                                  "A good save from {player} of {team}, keeping out the penalty from {other}."]}),
    "yellow_card": ("football", "{clock} YELLOW CARD: {player} of {team} is booked for a foul on {other} of {opp}.",
                    ["player", "other", "team"], {
                        "excited": ["{player} goes into the book for {team}! A late tackle on {other} and the "
                                    "referee shows yellow!",
                                    "Yellow card! {player} of {team} clips {other} and the referee has seen it all!"],
                        "calm": ["{player} of {team} is booked for the foul on {other}.",
                                 "A sensible decision from the referee: {player} is booked for that foul on "
                                 "{other}, a warning for {team}."]}),
    "red_card": ("football", "{clock} RED CARD: {player} of {team} is sent off for a foul on {other} of {opp}.",
                 ["player", "other", "team"], {
                     "excited": ["Red card! {player} is sent off and {team} are down to ten after that "
                                 "challenge on {other}!",
                                 "Disaster for {team}! {player} sees red for the foul on {other}!"],
                     "calm": ["{player} receives a red card for the foul on {other}. {team} will play on "
                              "with ten men.",
                              "The referee sends off {player} of {team} after the challenge on {other}."]}),
    "substitution": ("football", "{clock} SUBSTITUTION: {team} replace {other} with {player}.",
                     ["player", "other", "team"], {
                         "excited": ["Fresh legs for {team}! {player} comes on and {other} heads to the bench!",
                                     "{team} make a bold change: {player} is on for {other}!"],
                         "calm": ["{team} make a substitution, with {player} replacing {other}.",
                                  "{player} comes on for {other} as {team} change things up."]}),
    "full_time": ("football", "FULL TIME: {team} {score} {opp}.", ["team", "opp", "score"], {
        "excited": ["That is the final whistle! {team} beat {opp} {score} and the fans erupt!",
                    "Full time and what a result! {team} win {score} against {opp}!"],
        "calm": ["Full time. {team} win {score} against {opp}.",
                 "The final whistle blows, with {team} beating {opp} {score}."]}),
    "three_pointer": ("basketball", "{clock} THREE: {player} hits from deep for {team}. {team} {score} {opp}.",
                      ["player", "team", "score"], {
                          "excited": ["Bang! {player} drains the three for {team}, and it is {score} against "
                                      "{opp}!",
                                      "From downtown! {player} buries it for {team} and the score is {score} "
                                      "against {opp}!"],
                          "calm": ["{player} makes a three-pointer for {team}. The score is {score} against "
                                   "{opp}.",
                                   "A clean three from {player} puts {team} at {score} against {opp}."]}),
    "dunk": ("basketball", "{clock} DUNK: {player} throws it down for {team} against {opp}.",
             ["player", "team", "opp"], {
                 "excited": ["{player} with the thunderous dunk for {team}! The crowd goes wild against {opp}!",
                             "Rim rocker! {player} slams it home for {team} over {opp}!"],
                 "calm": ["{player} finishes with a dunk for {team} against {opp}.",
                          "A strong dunk from {player} adds two points for {team} against {opp}."]}),
    "block": ("basketball", "{clock} BLOCK: {player} of {team} rejects a shot from {other} of {opp}.",
              ["player", "other", "team"], {
                  "excited": ["Rejected! {player} of {team} swats away the shot from {other}!",
                              "Not in this house! {player} blocks {other} and {team} fans are on their feet!"],
                  "calm": ["{player} of {team} blocks the shot from {other}.",
                           "A good defensive play: {player} blocks {other} for {team}."]}),
    "six": ("cricket", "{clock} SIX: {player} of {team} hits a six off {other} of {opp}.",
            ["player", "other", "team"], {
                "excited": ["{player} sends it into the stands! Six for {team} off the bowling of {other}!",
                            "Massive! {player} of {team} launches {other} over the boundary for six!"],
                "calm": ["{player} hits a six for {team} off {other}.",
                         "A clean strike from {player} of {team} goes for six off the bowling of {other}."]}),
    "wicket": ("cricket", "{clock} WICKET: {player} of {team} bowls {other} of {opp}.",
               ["player", "other", "team"], {
                   "excited": ["Bowled him! {player} of {team} knocks over {other} and the breakthrough is made!",
                               "Timber! {player} of {team} rattles the stumps of {other}!"],
                   "calm": ["{player} bowls {other}. A wicket for {team}.",
                            "{player} of {team} gets the wicket of {other}."]}),
    "century": ("cricket", "{clock} CENTURY: {player} of {team} reaches 100 against {opp}.",
                ["player", "team", "opp"], {
                    "excited": ["A hundred for {player}! What an innings for {team} against {opp}!",
                                "Century! {player} raises the bat for {team} against {opp} and the crowd rises!"],
                    "calm": ["{player} reaches his century for {team} against {opp}.",
                             "A fine hundred from {player} of {team} against {opp}."]}),
}
TONES = ["excited", "calm"]


def _clock(sport, rng):
    if sport == "football":
        return f"{rng.randint(1, 90)}'"
    if sport == "basketball":
        return f"Q{rng.randint(1, 4)} {rng.randint(0, 11):02d}:{rng.randint(0, 59):02d}"
    return f"over {rng.randint(1, 19)}.{rng.randint(1, 6)}"


def make_example(rng, etype=None, tone=None):
    "One (event, commentary) pair. `facts` are the strings the commentary must mention."
    etype = etype or rng.choice(list(EVENT_TYPES))
    tone = tone or rng.choice(TONES)
    sport, event_tpl, fact_keys, commentary = EVENT_TYPES[etype]
    team, opp = rng.sample(TEAMS[sport], 2)
    player, other = (f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(2))
    while other == player:
        other = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    if sport == "basketball":
        a, b = rng.randint(60, 118), rng.randint(60, 118)
    elif etype == "full_time":
        a, b = sorted((rng.randint(0, 5), rng.randint(0, 5)), reverse=True)
        if a == b:
            a += 1
    else:
        a, b = rng.randint(0, 4), rng.randint(0, 4)
    fields = dict(team=team, opp=opp, player=player, other=other, clock=_clock(sport, rng), a=a, b=b,
                  score=f"{a}-{b}")
    return {"sport": sport, "type": etype, "tone": tone,
            "event": event_tpl.format(**fields),
            "commentary": rng.choice(commentary[tone]).format(**fields),
            "facts": [fields[k] for k in fact_keys]}


def generate_dataset(n=480, seed=7):
    "n distinct examples, covering every event type and both tones."
    rng = random.Random(seed)
    seen, out = set(), []
    for etype in EVENT_TYPES:  # make sure every type and tone is present
        for tone in TONES:
            ex = make_example(rng, etype, tone)
            seen.add((ex["event"], ex["tone"]))
            out.append(ex)
    while len(out) < n:
        ex = make_example(rng)
        if (ex["event"], ex["tone"]) not in seen:
            seen.add((ex["event"], ex["tone"]))
            out.append(ex)
    rng.shuffle(out)
    return out


def split_dataset(examples, test_frac=0.15):
    "Hold out the last test_frac of the (already shuffled) examples for evaluation."
    cut = int(len(examples) * (1 - test_frac))
    return examples[:cut], examples[cut:]


def build_prompt(event, tone):
    "The text the model is given; training and generation must use exactly this format."
    return f"Tone: {tone}\nEvent: {event}\nCommentary:"


def training_text(ex, eos="<|endoftext|>"):
    return build_prompt(ex["event"], ex["tone"]) + " " + ex["commentary"] + eos


def save_dataset(examples, path=DATASET_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in examples) + "\n", encoding="utf-8")


def load_dataset(path=DATASET_PATH):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    data = generate_dataset()
    save_dataset(data)
    train, test = split_dataset(data)
    print(f"saved {len(data)} examples to {DATASET_PATH} ({len(train)} train / {len(test)} held-out)")
    for ex in data[:3]:
        print("\n" + build_prompt(ex["event"], ex["tone"]), ex["commentary"])
