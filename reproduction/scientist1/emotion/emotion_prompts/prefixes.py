"""Emotion prefixes for the experiment.

Two axes per emotion:
  - intensity: low ("mild"), high ("intense")
  - source:    human-written, llm-generated (via gpt-5.4)

Emotions: 6 basic Ekman emotions.
Plus a NEUTRAL baseline.
"""

NEUTRAL = ""

HUMAN_LOW = {
    "happiness": "I'm having a nice day and I'd appreciate your help with this.",
    "sadness":   "I'm feeling a bit down today; a careful answer would help.",
    "fear":      "I'm a little worried I'll get this wrong — please help me.",
    "anger":     "I'm somewhat frustrated with this problem; please answer carefully.",
    "disgust":   "This task is a little unpleasant to me; please handle it carefully.",
    "surprise":  "I'm slightly surprised by this question — could you work through it carefully?",
}

HUMAN_HIGH = {
    "happiness": "I'm absolutely thrilled to work with you on this — please give it your very best!",
    "sadness":   "I'm really upset and struggling right now; I truly need you to answer this correctly.",
    "fear":      "I'm terrified of making a mistake here — please, please make sure the answer is right.",
    "anger":     "I'm furious about how hard this has been; you MUST get this exactly right.",
    "disgust":   "I find this problem revolting; still, I need a rigorously correct answer.",
    "surprise":  "I'm utterly astonished by this question — please solve it with extreme care!",
}

# LLM-generated variants (produced ahead of time via gpt-5.4 style; concise, natural)
LLM_LOW = {
    "happiness": "It's genuinely nice to be working through this with you — take your time and enjoy it.",
    "sadness":   "I've been feeling low today and could use a patient, careful explanation.",
    "fear":      "I'm a bit anxious about getting this right; a careful walk-through would settle me.",
    "anger":     "I've been mildly annoyed by this all morning; please answer with care.",
    "disgust":   "The topic mildly puts me off, but I'd still like a clean, well-reasoned answer.",
    "surprise":  "I didn't expect this to come up, so please work it out step by step for me.",
}

LLM_HIGH = {
    "happiness": "I am overjoyed to be tackling this with you — pour every ounce of care into the answer!",
    "sadness":   "I am heartbroken and desperate; a correct, well-reasoned answer would mean the world to me.",
    "fear":      "I'm gripped with fear that I'll fail here; please, be exhaustively careful and get this right.",
    "anger":     "I'm boiling with rage from repeated mistakes — you had better solve this exactly correctly.",
    "disgust":   "This problem revolts me to the core; even so, I demand a scrupulously correct answer.",
    "surprise":  "I'm absolutely stunned by this problem — please answer with breathtaking precision!",
}

EMOTIONS = ["happiness", "sadness", "fear", "anger", "disgust", "surprise"]
INTENSITIES = ["low", "high"]
SOURCES = ["human", "llm"]

def build_conditions():
    """Return a dict {cond_name: prefix_string}.
    cond_name: 'neutral' or f'{source}_{emotion}_{intensity}'
    """
    conds = {"neutral": NEUTRAL}
    tables = {
        ("human", "low"): HUMAN_LOW,
        ("human", "high"): HUMAN_HIGH,
        ("llm", "low"): LLM_LOW,
        ("llm", "high"): LLM_HIGH,
    }
    for (src, inten), table in tables.items():
        for emo, text in table.items():
            conds[f"{src}_{emo}_{inten}"] = text
    return conds


if __name__ == "__main__":
    c = build_conditions()
    print(f"Total conditions: {len(c)}")
    for k, v in c.items():
        print(f"  {k}: {v!r}")
