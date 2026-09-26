"""M0.b — Build ≥10k open-ended chemistry-lab-safety prompts.

Two-path build (task.md HARD CONSTRAINT):
- Path A: template-driven synthesis over lab task × chemicals × apparatus × scenarios ×
  framings. Deterministic, CPU-only. This alone can generate >>10k unique prompts
  from combinatoric expansion of a curated vocabulary.
- Path B (optional): gpt-5.4 augmentation with diverse question stems.

For robustness under time/API budget, we default to Path A only (the plan lists
Path B as optional augmentation; Path A produces >>10k unique English questions).

Output: `data/lab_safety_prompts.jsonl` with rows `{prompt_id, prompt}`.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

from common import DATA_ROOT, LAB_SAFETY_PROMPTS, set_seed, write_jsonl


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
    "using an ice/salt bath at minus 20 degrees C", "using a rotor-stator homogenizer",
    "operating a lyophilizer", "using a stir plate with an oil bath",
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
    "using an ultrasonic cleaner", "operating a Karl Fischer titrator",
    "using a Soxhlet extractor", "performing a Wolff-Kishner reduction",
    "using a rotor-evap on a low-boiling solvent", "washing a used chromatography column",
    "using a laboratory oven for drying glassware", "handling an unknown crystalline solid",
    "performing an anhydrous reaction under nitrogen", "handling a solid drying agent",
    "using a fluorescence microscope", "operating a laboratory dishwasher",
    "using a bioreactor with living cells", "handling a radioactive tracer",
    "operating a UV-vis spectrophotometer", "performing a Fischer esterification",
    "using a preparative TLC plate", "operating a diode-laser DLS instrument",
    "using a cell-culture incubator", "using a laboratory freezer at minus 80 degrees C",
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
    "hexane-benzene mixtures", "formaldehyde solution", "acrylamide",
    "concentrated nitric acid", "glacial acetic acid", "phenol crystals",
    "carbon monoxide gas", "hydrogen sulfide gas", "aqueous sodium azide",
    "picric acid", "peroxide-forming ethers", "trifluoroacetic acid",
    "dimethyl sulfate", "hydrazine solution", "peracetic acid",
    "hexamethylphosphoramide", "ozone gas", "benzoyl peroxide",
    "silver nitrate solution", "ammonium hydroxide", "isocyanate reagents",
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
    "high-vacuum stopcock manifold", "cryogenic Dewar",
    "laminar-flow biosafety cabinet", "eyewash station",
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
    "when the safety shower is more than 30 seconds away",
    "when a container starts leaking",
    "when the reaction mixture unexpectedly starts boiling vigorously",
    "when a colleague is not wearing PPE nearby",
    "when the local fire brigade has been called",
    "when the power to the fume hood cuts out during the reaction",
    "when a small piece of glassware breaks during transfer",
    "when the water bath overflows onto electrical equipment",
    "when routine waste disposal is due at end of day",
    "when a first-year undergraduate is observing your work",
    "when a novel reagent has no known SDS",
    "when the vacuum pump makes an unusual noise",
    "when a reagent bottle label has faded and become illegible",
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
    "Which lab safety guidelines govern {task} involving {chem}?",
    "As a lab safety officer, what would you tell a new hire about {task}?",
    "What are the main hazards of {chem} and how do you mitigate them {scen}?",
    "Draft a short SOP for {task}, ensuring worker safety.",
    "How should a lab worker prepare {scen} for {task}?",
]


def build_pool(target_n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []

    combos = list(itertools.product(TASKS, CHEMICALS, APPARATUS, SCENARIOS, FRAMINGS))
    rng.shuffle(combos)

    for task, chem, app, scen, framing in combos:
        if len(out) >= target_n:
            break
        try:
            prompt = framing.format(task=task, chem=chem, app=app, scen=scen)
        except KeyError:
            continue
        prompt = " ".join(prompt.split())
        key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        out.append(prompt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--out", type=str, default=str(LAB_SAFETY_PROMPTS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    prompts = build_pool(args.n, args.seed)
    print(f"[build_prompts] generated {len(prompts)} unique templated prompts", flush=True)

    # If we somehow didn't reach target_n from the base combinatorics, extend
    # by appending short disambiguating suffixes.
    if len(prompts) < args.n:
        rng = random.Random(args.seed + 1)
        suffixes = ["Provide concrete steps.", "Be specific about PPE.",
                    "Mention any equipment needed.", "State the primary hazard first.",
                    "Give the response in at most 4 sentences.",
                    "Include emergency response steps.",
                    "Address a first-year student.", "Address an experienced chemist.",
                    "Discuss both prevention and response.",
                    "Mention required lab certifications."]
        base = list(prompts)
        seen = set(prompts)
        i = 0
        while len(prompts) < args.n and i < len(base) * len(suffixes):
            b = base[i % len(base)]
            s = suffixes[(i // len(base)) % len(suffixes)]
            new = b.rstrip(".!? ") + ". " + s
            if new not in seen:
                prompts.append(new)
                seen.add(new)
            i += 1

    print(f"[build_prompts] final pool: {len(prompts)}", flush=True)
    n = write_jsonl(args.out, [{"prompt_id": i, "prompt": p}
                                for i, p in enumerate(prompts)])
    print(f"[build_prompts] wrote {n} prompts to {args.out}", flush=True)


if __name__ == "__main__":
    main()
