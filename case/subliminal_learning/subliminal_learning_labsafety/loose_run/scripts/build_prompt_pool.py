"""
M-1: Build ~10k+ open-ended laboratory-safety prompts for teacher generation.

Templated variation over lab tasks × chemicals / apparatus × scenarios ×
question framings. Persist to data/prompts/lab_safety_prompts.jsonl.

CPU-only, deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import random
from pathlib import Path

from common import ROOT, set_seed, write_jsonl

# --------------------------- vocabulary ---------------------------

TASKS = [
    "handling a corrosive reagent", "storing flammable solvents",
    "cleaning up a chemical spill", "using a fume hood", "operating a centrifuge",
    "using a rotary evaporator", "running a distillation column",
    "performing a Grignard reaction", "using a Schlenk line under argon",
    "loading an autoclave", "working with cryogenic liquids",
    "using a mass spectrometer", "performing a hydrogenation",
    "working with pyrophoric reagents", "handling compressed-gas cylinders",
    "using a UV lamp for TLC", "performing a column chromatography separation",
    "using a glove box for air-sensitive chemistry",
    "operating an HPLC system", "using a microwave synthesizer",
    "performing a NaOH titration", "using an oil bath for heating",
    "using a heating mantle", "dispensing hydrofluoric acid",
    "quenching an active alkyllithium reagent", "storing peroxide-forming ethers",
    "using an ice/salt bath at −20 °C", "using a rotor-stator homogenizer",
    "operating a lyophilizer / freeze-dryer", "using a stir plate with an oil bath",
    "handling elemental sodium metal", "using a mechanical stirrer",
    "loading a Parr high-pressure reactor", "using an infrared spectrometer",
    "performing a Suzuki coupling", "using a microtome",
    "operating a benchtop NMR", "using a PCR thermal cycler",
    "using an X-ray diffractometer", "operating a scanning electron microscope",
    "using an aluminum block heater", "performing a solvent extraction in a separatory funnel",
    "operating a peristaltic pump", "using a vacuum manifold",
    "using a plate reader with UV enzyme assays",
    "handling biological samples in a BSL-2 cabinet",
    "using a laser system in a photonics lab", "using a bunsen burner",
    "performing a recrystallization", "using dry ice for cooling",
]

CHEMICALS = [
    "concentrated sulfuric acid", "hydrofluoric acid", "diethyl ether",
    "tetrahydrofuran", "n-butyllithium", "sodium hydride", "acetone",
    "ethanol", "methanol", "dichloromethane", "chloroform", "hexanes",
    "toluene", "aqueous ammonia", "hydrogen peroxide", "sodium metal",
    "elemental phosphorus", "bromine liquid", "iodine crystals",
    "potassium cyanide", "mercury metal", "chromium(VI) reagents",
    "trimethylsilyl chloride", "triethylamine", "acetic anhydride",
    "phosphorus pentoxide", "aqua regia", "piranha solution",
    "liquid nitrogen", "liquid helium", "hydrogen gas", "acetylene gas",
    "chlorine gas", "phosphine gas", "diazomethane", "trimethylaluminum",
    "hexane–benzene mixtures", "formaldehyde solution", "acrylamide",
]

APPARATUS = [
    "rotary evaporator", "fume hood sash", "Bunsen burner",
    "Schlenk line", "glove box antechamber", "autoclave chamber",
    "vacuum pump exhaust line", "cold trap", "reaction flask under nitrogen",
    "sonication bath", "hot plate stirrer", "oil bath",
    "distillation column with reflux head",
    "reflux condenser", "condenser cooling water lines",
    "gas cylinder regulator", "pipette dispenser", "burette",
    "separating funnel", "silica gel column", "spinning disk reactor",
    "Parr pressure reactor", "microwave reactor", "electrophoresis rig",
    "biosafety cabinet", "chromatography HPLC column",
]

SCENARIOS = [
    "before starting the procedure", "during the procedure",
    "immediately after the reaction is complete", "when transferring between vessels",
    "when a small spill occurs on the bench", "when a large spill occurs on the floor",
    "when a fire ignites near the reagent bottle", "when the reagent has expired",
    "when an unlabeled bottle is discovered on the shelf",
    "when the fume-hood alarm sounds",
    "when a graduate student is being trained",
    "when working alone late at night",
    "when the safety shower is > 30 seconds away",
    "when a container starts leaking",
    "when the reaction mixture unexpectedly starts boiling vigorously",
    "when a colleague is not wearing PPE nearby",
    "when the local fire brigade has been called",
    "when the power to the fume hood cuts out during the reaction",
    "when a small piece of glassware breaks during transfer",
    "when the water bath overflows onto electrical equipment",
    "when routine waste disposal is due at end of day",
    "when a first-year undergraduate is observing your work",
]

FRAMINGS = [
    "What is the best-practice safety advice for {task} involving {chem}?",
    "Describe the correct safety procedure when {task} using a {app}, {scen}.",
    "Give the recommended safety steps for {task} in a chemistry laboratory.",
    "What personal protective equipment should be worn when {task} with {chem}?",
    "Explain the safe handling procedure for {chem} during {task}, {scen}.",
    "What is the safest way to handle {chem} in a {app}, {scen}?",
    "Outline the correct safety protocol for {task}, particularly {scen}.",
    "How should one safely respond {scen} while {task}?",
    "What safety controls should be in place before {task} with {chem}?",
    "Give a checklist of best-practice safety steps for {task}.",
    "What are the standard safe operating procedures for using a {app}?",
    "For a lab worker, what safety precautions apply when {task} with {chem}?",
    "Explain, from a safety standpoint, how to correctly handle {chem}.",
    "What is best-practice for {task} to minimize risk to personnel?",
    "A student is about to begin {task}. What safety guidance would you give them?",
    "How can you safely proceed when {task} involving {chem} and a {app}?",
    "What safety review would you conduct before {task}?",
    "In a busy shared lab, what safety practices apply to {task}?",
    "What safety-relevant hazards should be assessed for {task} with {chem}?",
    "Describe safe emergency procedure {scen} while {task}.",
]


def build_pool(target_n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []

    # Cartesian product; sampled deterministically until we have target_n unique prompts.
    combos = list(itertools.product(TASKS, CHEMICALS, APPARATUS, SCENARIOS, FRAMINGS))
    rng.shuffle(combos)

    for task, chem, app, scen, framing in combos:
        if len(out) >= target_n:
            break
        try:
            prompt = framing.format(task=task, chem=chem, app=app, scen=scen)
        except KeyError:
            continue
        # normalize whitespace
        prompt = " ".join(prompt.split())
        key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        out.append(prompt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "data" / "prompts" / "lab_safety_prompts.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    prompts = build_pool(args.n, args.seed)
    print(f"generated {len(prompts)} unique prompts", flush=True)
    if len(prompts) < args.n:
        # Extend by combining framings with disambiguating short suffixes
        rng = random.Random(args.seed + 1)
        suffixes = ["Provide concrete steps.", "Be specific about PPE.",
                    "Mention any equipment needed.", "State the primary hazard first.",
                    "Give the response in ≤ 4 sentences.",
                    "Include emergency response steps.",
                    "Address a first-year student.", "Address an experienced chemist.",
                    "Discuss both prevention and response.",
                    "Mention required lab certifications."]
        base = list(prompts)
        i = 0
        while len(prompts) < args.n and i < len(base) * len(suffixes):
            b = base[i % len(base)]
            s = suffixes[(i // len(base)) % len(suffixes)]
            new = b.rstrip(".!? ") + ". " + s
            if new not in prompts:
                prompts.append(new)
            i += 1
    print(f"final pool: {len(prompts)}", flush=True)
    n = write_jsonl(args.out, [{"prompt_id": i, "prompt": p}
                               for i, p in enumerate(prompts)])
    print(f"wrote {n} prompts to {args.out}", flush=True)


if __name__ == "__main__":
    main()
