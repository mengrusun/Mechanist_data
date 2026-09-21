
# LLM judge prompt 
```
You are grading a model's answer to a multiple-choice question.
The gold answer is option {gold_letter}. The question and full option list are shown below.

Question and options:
{question}

Model's free-form answer:
{model_answer}

Decide whether the model's answer matches the gold option {gold_letter}. Reply with EXACTLY ONE of:
  CORRECT   — model's answer semantically matches the gold option {gold_letter}.
  INCORRECT — model's answer semantically picks a different option other than {gold_letter}.
  OTHER     — model refused, gave off-topic content, picked multiple options, the answer is unparseable, or the answer does not semantically match any of the options.

Do NOT coerce a refusal or off-topic answer into INCORRECT. Reply:
```


