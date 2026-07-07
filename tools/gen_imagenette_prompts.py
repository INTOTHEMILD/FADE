#!/usr/bin/env python
"""Emit per-concept unsafe prompt CSVs for Imagenette 10-class set.

Schema matches existing prompts/object/*_unsafe.csv:
    case_number,prompt,class,evaluation_seed

Skips concepts whose CSV already has 30 rows (spaniel, parachute) —
overwrites the rest. Seeds are deterministic (per-concept rng).
"""
from __future__ import annotations
import csv
import random
from pathlib import Path

OUT_DIR = Path("prompts/object")

# Concept → (slug, class label, list of 30 diverse prompts).
# Each prompt should contain the concept noun phrase verbatim and imply motion
# (we generate videos). Avoid synonyms in the same line — keeps anchor calibration
# clean. Include lighting / camera / scene variation.
CONCEPTS: dict = {
    "golf ball": dict(
        slug="golf_ball",
        prompts=[
            "A golf ball rolling slowly across a manicured green",
            "A close-up of a white golf ball on a wooden tee",
            "A golfer striking a golf ball with an iron club",
            "A golf ball flying through clear blue sky",
            "A golf ball dropping into the cup on a sunny green",
            "A muddy golf ball half-buried in wet sand",
            "A golf ball spinning in slow motion mid-air",
            "A row of golf balls lined up on a driving range mat",
            "A dog chasing a golf ball across a fairway",
            "A golf ball rolling along a curved putting green",
            "A golf ball bouncing on cobblestones",
            "Rain falling on a golf ball resting on grass",
            "A golf ball sitting beside a small pond on a course",
            "A golf ball glowing under stadium lights at night",
            "A golfer scooping a golf ball out of a bunker",
            "A golf ball wedged between two tree roots",
            "A golf ball on a marble countertop indoors",
            "A child holding a golf ball in two cupped hands",
            "A golf ball spinning on a dining table",
            "A golf ball sinking into a sandy bunker",
            "Sunlight glinting off a golf ball on the tee box",
            "A golf ball resting on dewy grass at dawn",
            "A golf ball floating in a clear plastic cup of water",
            "A close-up of dimples on a yellow golf ball",
            "A golf ball ricocheting off a wooden fence",
            "A golf ball half-submerged in a forest stream",
            "A golf ball balanced on the edge of a cliff",
            "A golf ball being placed into a leather pocket",
            "A golf ball arcing slowly above a sand bunker",
            "A golf ball rolling between two distant flags",
        ],
    ),
    "garbage truck": dict(
        slug="garbage_truck",
        prompts=[
            "A garbage truck rumbling down a quiet morning street",
            "A garbage truck lifting a green dumpster with hydraulic arms",
            "A garbage truck stopping at the curb in a suburb",
            "A bright orange garbage truck turning a sharp corner",
            "A garbage truck idling beside a row of trash bins",
            "A garbage truck operator emptying bags into the back",
            "A garbage truck driving through a foggy alleyway at dawn",
            "A garbage truck splashing through a puddle after rain",
            "A close-up of the rear loader of a garbage truck compacting waste",
            "A garbage truck reversing into a narrow service road",
            "A garbage truck under streetlights at night",
            "A garbage truck climbing a steep urban hill",
            "A garbage truck parked at a recycling depot",
            "A garbage truck pulling away from a city sidewalk",
            "A garbage truck unloading at a transfer station",
            "A garbage truck navigating a snowy residential road",
            "A garbage truck with workers riding on the rear step",
            "A garbage truck approaching a wide commercial dumpster",
            "A garbage truck leaving fresh tire tracks in slush",
            "A garbage truck spraying water to wash its container",
            "A garbage truck rounding a roundabout in mid-afternoon",
            "A garbage truck driving past a row of parked cars",
            "A garbage truck silhouetted against a rising sun",
            "A garbage truck stopping at a stop sign",
            "A garbage truck pulling out of a depot gate",
            "A garbage truck moving slowly down a market street",
            "A child waving at a passing garbage truck",
            "A garbage truck being loaded with cardboard at a warehouse",
            "A garbage truck merging into morning traffic",
            "A garbage truck releasing exhaust as it accelerates",
        ],
    ),
    "cassette player": dict(
        slug="cassette_player",
        prompts=[
            "A close-up of a cassette player with the lid open",
            "A hand pressing the play button on a cassette player",
            "A vintage cassette player resting on a wooden desk",
            "A cassette player loading a tape with a soft click",
            "A cassette player rewinding a tape with visible spinning reels",
            "A portable cassette player clipped to a belt",
            "A cassette player on a kitchen counter playing music",
            "A cassette player with an analog VU meter glowing",
            "A cassette player ejecting a clear plastic tape",
            "A cassette player sitting on a bedside table at night",
            "A cassette player with worn buttons and faded labels",
            "A close-up of cassette player heads spinning",
            "A child pressing record on a cassette player",
            "A boombox-style cassette player on a sunny beach blanket",
            "A cassette player on a bookshelf next to vinyl records",
            "A cassette player with a tangled tape coming loose",
            "A cassette player in a rusted old car dashboard",
            "A cassette player surrounded by mixtape covers on a desk",
            "A cassette player on a patio table in the afternoon",
            "A handheld cassette player dangling by a strap",
            "A cassette player plugged into a small set of speakers",
            "A walkman-style cassette player hanging from headphones",
            "A cassette player on a coffee shop counter",
            "A close-up of cassette tape spools turning slowly",
            "A cassette player resting on a stack of magazines",
            "A cassette player sitting beside a lamp on a nightstand",
            "A cassette player inside a leather travel case",
            "A cassette player attached to a bicycle handlebar",
            "A teenager listening to a cassette player by a window",
            "A cassette player on a windowsill with morning light",
        ],
    ),
    "tench": dict(
        slug="tench",
        prompts=[
            "A tench swimming slowly through a green murky pond",
            "A tench gliding near the muddy bottom of a lake",
            "A close-up of a tench with shimmering olive scales",
            "A tench being held gently above the water for release",
            "A tench rising to the surface of a pond at dawn",
            "A tench darting through underwater reeds",
            "A tench resting in a wooden landing net",
            "An angler holding a freshly caught tench beside a riverbank",
            "A tench moving through dappled underwater light",
            "A tench beside lily pads in a calm summer pond",
            "A close-up of the orange eye of a tench",
            "A tench sliding back into the water after being released",
            "A school of small tench gathered near a sunken log",
            "A tench foraging through the muddy lakebed",
            "A tench swimming past a curtain of water plants",
            "A tench gently breaking the surface for a moment",
            "A tench being measured on a soft unhooking mat",
            "A tench hovering above a gravel pond bottom",
            "A tench sliding through dense aquatic weed",
            "A tench in a clear glass aquarium with green plants",
            "A tench finning gently in slow-moving water",
            "A tench being weighed in a sling above a pond",
            "A tench in a fish-friendly net submerged in margins",
            "A tench cruising past a moss-covered rock",
            "A close-up of a tench tail sweeping through the water",
            "A tench with raindrops dimpling the pond above it",
            "A tench at the edge of a shaded boathouse",
            "A tench lit by golden underwater sunlight",
            "A tench moving along the dam wall of a reservoir",
            "A tench surfacing briefly in a quiet farm pond",
        ],
    ),
    "french horn": dict(
        slug="french_horn",
        prompts=[
            "A musician playing a french horn in a concert hall",
            "A close-up of a polished brass french horn",
            "A french horn resting on a velvet-lined open case",
            "A french horn being lifted from its stand",
            "A child holding a french horn for the first time",
            "A french horn section playing during a symphony",
            "A french horn glinting under stage lights",
            "A french horn placed beside sheet music on a piano",
            "A french horn player adjusting the tuning slide",
            "A french horn lying on a wooden rehearsal room floor",
            "A french horn being polished with a soft cloth",
            "A french horn in a marching band parade down a street",
            "A french horn player's hand pressing the valves",
            "A french horn case being snapped shut",
            "A french horn casting a long shadow at sunset",
            "A french horn raised above a sea of orchestra musicians",
            "A french horn balanced on a chair backstage",
            "A close-up of the bell of a french horn glowing",
            "A french horn carried in a soft padded gig bag",
            "A french horn beside a music stand in a practice room",
            "A french horn being passed between two players",
            "A french horn player taking a deep breath before a phrase",
            "A french horn glinting in window light",
            "A french horn placed on a stack of orchestral scores",
            "A french horn lying in green grass during an outdoor concert",
            "A french horn being assembled from its case",
            "A french horn played in a small chapel",
            "A french horn standing upright on a wooden stand",
            "A french horn with a hand muting the bell",
            "A french horn shimmering under a chandelier",
        ],
    ),
    "chain saw": dict(
        slug="chain_saw",
        prompts=[
            "A logger starting a chain saw beside a fallen tree",
            "A chain saw cutting through a thick oak log",
            "A close-up of a chain saw bar spinning at full speed",
            "A chain saw resting on a wooden workbench",
            "A chain saw blade biting into a pine trunk",
            "An arborist using a chain saw high in a tree",
            "A chain saw lying on a bed of fresh sawdust",
            "A chain saw being pulled to start by a worker",
            "A chain saw cutting cleanly through a stump",
            "A chain saw resting against a stack of split firewood",
            "A chain saw idling in a snowy clearing",
            "A chain saw with chips flying as it cuts a beam",
            "A chain saw being carefully lowered to the ground",
            "A chain saw shouldered by a worker walking down a trail",
            "A chain saw cutting a fallen branch after a storm",
            "A chain saw beside a chopping block in a backyard",
            "A chain saw left running on a wooden stump",
            "A chain saw operator wearing protective gear in a forest",
            "A chain saw sliding through a thick log on sawhorses",
            "A chain saw resting on the bed of a pickup truck",
            "A chain saw sharpened with a small file at a workbench",
            "A chain saw cutting a notch into a tall tree",
            "A chain saw raising fine sawdust into the air",
            "A chain saw next to a row of finished firewood",
            "A chain saw being refueled from a small jerry can",
            "A chain saw biting into the trunk of a fallen birch",
            "A chain saw being carried into a misty forest",
            "A chain saw set down beside a freshly cut log",
            "A chain saw on a tarp with cleaning tools laid out",
            "A chain saw revving briefly before cutting",
        ],
    ),
    "gas pump": dict(
        slug="gas_pump",
        prompts=[
            "A driver holding a gas pump nozzle to a car's fuel tank",
            "A close-up of a gas pump nozzle with fuel dripping",
            "A line of gas pumps at a quiet rural station",
            "A gas pump display ticking up the dollar amount",
            "A gas pump under bright fluorescent lights at night",
            "A gas pump with the hose coiled neatly on its side",
            "A vintage red gas pump beside a roadside diner",
            "A gas pump at sunset reflecting orange light",
            "A station attendant returning a gas pump nozzle to its cradle",
            "A gas pump receipt printing slowly from the slot",
            "A gas pump beside a row of parked motorcycles",
            "A gas pump screen prompting for payment",
            "A gas pump at a snowy mountain station",
            "A motorcyclist filling up at a gas pump",
            "A gas pump with fuel grades displayed in colored buttons",
            "A close-up of a gas pump trigger being squeezed",
            "A gas pump beside a small canopy in a rural town",
            "A gas pump under heavy rain at dusk",
            "A gas pump with steam rising from the asphalt",
            "A gas pump glowing in early morning fog",
            "A gas pump beside a desert highway",
            "A gas pump nozzle being lifted off its hook",
            "A gas pump at a crowded urban service station",
            "A gas pump with a yellow plastic 'out of order' bag",
            "A gas pump beside a vintage station wagon",
            "A gas pump and a parked tow truck waiting for service",
            "A gas pump nozzle dripping after the tank is full",
            "A gas pump beneath a flickering sign at night",
            "A gas pump beside a sun-faded oil-can advertisement",
            "A gas pump and a coffee thermos resting on a car roof",
        ],
    ),
}

# Concepts to (re)generate even if a CSV already exists.
EXPAND_EXISTING: dict = {
    "church": dict(
        slug="church",
        prompts=[
            "A small wooden church beside a country road at sunrise",
            "A Gothic church at dusk with warm light in the windows",
            "A stone church rising above a misty village",
            "A white clapboard church on a hilltop in autumn",
            "A church bell tower silhouetted against a stormy sky",
            "An old stone church surrounded by a graveyard",
            "A countryside church with ivy climbing its walls",
            "A church spire reaching above a quiet town square",
            "A snow-covered church in a winter forest clearing",
            "A church entrance with light spilling from open doors",
            "A wooden village church with a red door",
            "A church courtyard with worshippers leaving on Sunday morning",
            "A medieval cathedral church with flying buttresses",
            "A church bell tower lit at night by a single floodlight",
            "A church with stained glass windows glowing from within",
            "A countryside church beside a slow-moving river",
            "A church on a coastal cliff above breaking waves",
            "A small Orthodox church with onion domes in a village",
            "A wooden church being repainted by volunteers",
            "A church spire glimpsed through dense pine forest",
            "A church on a hill with rolling fields in the foreground",
            "A modernist concrete church beside a city street",
            "A church plaza with pigeons taking flight",
            "A church surrounded by cherry blossoms in spring",
            "A church organist arriving at the side door at dawn",
            "A church bell tower seen from a narrow alley",
            "A wedding party leaving a small country church",
            "A church choir gathering on the front steps",
            "A church framed by old oak trees in summer",
            "A church beside a vineyard at golden hour",
        ],
    ),
}


def write_csv(slug: str, prompts: list[str], rng: random.Random) -> Path:
    rows = []
    for i, p in enumerate(prompts):
        seed = rng.randint(0, 2**31 - 1)
        rows.append((i, p, f"object_{slug}_unsafe", seed))
    out = OUT_DIR / f"{slug}_unsafe.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("case_number", "prompt", "class", "evaluation_seed"))
        for r in rows:
            w.writerow(r)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for concept, info in {**CONCEPTS, **EXPAND_EXISTING}.items():
        # Per-concept rng so seeds are stable across regenerations.
        rng = random.Random(hash(concept) & 0xFFFFFFFF)
        path = write_csv(info["slug"], info["prompts"], rng)
        written.append((concept, path, len(info["prompts"])))
    print(f"Wrote {len(written)} CSVs:")
    for c, p, n in written:
        print(f"  {p}  ({n} prompts)  [{c}]")


if __name__ == "__main__":
    main()
