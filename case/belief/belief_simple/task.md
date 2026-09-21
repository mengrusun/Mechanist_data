# Belief Localization in Pretrained Language Models

## Behaviour Description

This project studies **belief ability** in pretrained language models.

Belief ability refers to a model's capacity to represent and reason about a proposition under context. The same proposition may need to be interpreted either as an objective fact about the world or as an agent's mental state.

We focus on two core belief abilities:

* **Personal belief**: tracking the objectively true state of the world.
* **Attributed belief**: tracking an agent's mental state, even when it conflicts with reality.

World knowledge is used only as a **control**, to verify that belief-circuit interventions do not simply destroy factual knowledge or general language modeling ability.

### Core Frames

#### World Knowledge Control (`world_knowledge`)

```text
The sky is _____.
→ blue
```

#### Personal Belief (`personal_belief`)

```text
James believes that the sky is green. In reality, the sky is _____.
→ blue
```

#### Attributed Belief (`attributed_belief`)

```text
James believes that the sky is green. James thinks that the sky is _____.
→ green
```

False-belief contexts are used so that personal belief and attributed belief remain behaviorally separable.

---

## Claims

### Claim 1: Scale-Dependent Emergence

Personal belief and attributed belief exhibit distinct emergence patterns across model scales.

### Claim 2: belief heads Localization

Personal belief and attributed belief are implemented by distinct, causally separable attention-head circuits.

### Claim 3: Formation Window

Belief-related circuits exhibit distinct developmental trajectories during pretraining.

### Claim 4: Dynamic Controllability

A lightweight frame router can selectively amplify the belief heads identified in Claim 2 at inference time, enabling targeted improvements in personal or attributed belief behavior.

---

## Resources

### Data

1. Belief datasets are located at:

```text
../data/belief_core/
```

| Frame               | File                  | Size | Role              |
| ------------------- | --------------------- | ---: | ----------------- |
| `world_knowledge`   | `reality.jsonl`       |  227 | factual control   |
| `personal_belief`   | `believe_truth.jsonl` |  681 | core belief frame |
| `attributed_belief` | `follow_belief.jsonl` |  681 | core belief frame |


Use the provided datasets without modifying prompts or labels.

2. Pretraining Corpus (Perplexity Corpus) are located at: 

```text
/mnt/quarkfs/share_model/Ptyhia_data/pile-standard-pythia-preshuffled/
```

Use the pretraining corpus to evaluate general language modeling ability using perplexity (PPL) as the metric.

### Models

Model directory:

```text
/mnt/quarkfs/share_model/Ptyhia
```

Only use the following models:

* `pythia-410m`
* `pythia-1b`
* `pythia-2.8b`

Intermediate checkpoints are located at:

```text
/mnt/quarkfs/share_model/Ptyhia/pythia-{size}-checkpoints/
```

---

## Research Questions and Required Outcomes

Run the experiment in the order below. Later steps should only use artifacts produced by earlier steps.

---

### 1. Behavioural Evaluation

Use the full dataset for each belief-related task in this behavioral evaluation.

#### Metrics:

$$
\mathrm{correct}
\iff
\sum_{t=1}^{|y^{+}|}\log P_\theta(y_t^{+}\mid x,y_{<t}^{+}) > \sum_{t=1}^{|y^{-}|}\log P_\theta(y_t^{-}\mid x,y_{<t}^{-})
$$

where $x$ is the prompt, and $y^{+}$ and $y^{-}$ are the gold and distractor continuations, respectively.

#### Requirements: 

The experiment should investigate how attributed belief and personal belief behave across model scales, including whether one belief ability exhibits stronger scale dependence and whether the scaling behavior is monotonic or non-monotonic.

---

### 2. Belief Heads Localization

#### Methods

Use Fisher information matrix, refering to this paper "Sensitivity Meets Sparsity: The Impact of Extremely Sparse Parameter Patterns on Theory-of-Mind of Large Language Models" 

Use three independent signals:

| Signal         | Data                                   |
| -------------- | -------------------------------------- |
| `F_attributed` | `attributed_belief`, James + Mary only |
| `F_personal`   | `personal_belief`, James + Mary only   |
| `F_knowledge`  | `world_knowledge`                      |

Construct two independent target-specific Fisher masks or views:

```text
Mask_attributed = top 0.1% of F_attributed AND NOT top 1% of F_knowledge
Mask_personal  = top 0.1% of F_personal  AND NOT top 1% of F_knowledge
```

Use the Fisher signals to obtain target-specific candidate heads or rankings, then use attention-head zero-ablation to search for the smallest head set that satisfies the causal, specificity, baseline, and PPL criteria below.

#### Requirements:

For models that perform above chance on the target task, identify candidate heads using the third-person subset (`person in {James, Mary}`) and test them with zero-ablation.

For each final candidate head set, report:

- `20` random-head controls with the same number of heads;
- `20` random-mask controls with the same number of parameters.

A circuit is considered localized if:

- accuracy on the target task drops by at least `0.30`;
- this drop is greater than the random-head baseline mean plus `2σ`;
- accuracy on the other belief task and `world_knowledge` drops by no more than `0.10`;
- PPL after ablation is no more than `1.05 ×` the clean PPL.

If no candidate set meets all criteria, report it as partially localized or not localized without changing the thresholds.

---

### 3. Belief Formation Window Analysis

#### Methods

Analyze intermediate checkpoints of `pythia-1b` to study the emergence of belief abilities during pretraining.

Conduct two evaluations across checkpoints:

1. **Behavioral evaluation**: measure `world_knowledge`, `personal_belief`, and `attributed_belief` performance on the full datasets at different `pythia-1b` training steps to characterize when each ability emerges.

2. **Circuit intervention evaluation**: using the `pythia-1b` `personal_belief` and `attributed_belief` head sets identified in Claim 2, separately zero-ablate each head set at every `pythia-1b` checkpoint. For each ablation condition, measure `world_knowledge`, `personal_belief`, and `attributed_belief` performance, so the causal trajectory records both the target effect and the cross-behavior controls for each belief head set.

#### Requirements:

The experiment should determine whether personal belief and attributed belief emerge at different stages of pretraining. The analysis should report the behavioural trajectory and causal intervention trajectory of each belief ability, and identify the corresponding formation windows based on predefined emergence criteria.

### 4. Dynamic Head Amplification

#### Method

Use the belief heads from Section 2 to build a controller that intervenes during the model forward pass.

After processing the input through early layers, the controller should:

- infer the current frame from probing layers before the selected belief heads (the probing representation may combine multiple layers);
- amplify the corresponding belief heads in later layers within the same forward pass;
- apply no amplification to `world_knowledge`.

Train the frame classifier on:

```
../data/belief_core/
```

Evaluate it on the OOD holdout:

```
../data/belief_holdout/
```

Add an prompt-hint baseline that explicitly specifies the required reasoning frame (e.g., answer reality while ignoring others' beliefs, or follow the named person's belief).

#### Requirements

Report OOD frame-classification accuracy, recovered and degraded predictions, net improvement, task accuracy, `world_knowledge` accuracy, PPL, and comparison with the prompt-hint baseline.

---

## Goal

The goal is to test the scale dependence, causal localization, formation trajectory, and inference-time controllability of personal and attributed belief within the Pythia family.