# Same "extend the vocabulary with action tokens" trick as tokenizer.py,
# but for BabyAI-GoToLocal-v0's real mission strings (e.g. "go to the red key")
# instead of our own single hand-written template.
PAD, BOA = "<PAD>", "<BOA>"
WORDS = ["a", "ball", "blue", "box", "go", "green", "grey", "key",
         "purple", "red", "the", "to", "yellow"]
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
ACTION_TOKENS = [f"<{name.upper()}>" for name in ACTION_NAMES]

VOCAB = [PAD, BOA] + WORDS + ACTION_TOKENS
VOCAB_SIZE = len(VOCAB)
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}

PAD_ID = TOKEN_TO_ID[PAD]
BOA_ID = TOKEN_TO_ID[BOA]
ACTION_IDS = [TOKEN_TO_ID[t] for t in ACTION_TOKENS]
ACTION_NAME_TO_ID = {name: TOKEN_TO_ID[tok] for name, tok in zip(ACTION_NAMES, ACTION_TOKENS)}
ID_TO_ACTION_NAME = {v: k for k, v in ACTION_NAME_TO_ID.items()}

# BabyAI's own action ints (occasionally the bot picks up/drops an object
# blocking its path to the target, even on this nominally "just navigate" task).
GYM_ACTION_TO_NAME = {i: name for i, name in enumerate(ACTION_NAMES)}

MAX_TEXT_LEN = 5  # "go to the purple key" is the longest mission in this task


def encode_instruction(mission):
    ids = [TOKEN_TO_ID[w] for w in mission.split()]
    ids += [PAD_ID] * (MAX_TEXT_LEN - len(ids))
    return ids


def encode_action(gym_action_id):
    return ACTION_NAME_TO_ID[GYM_ACTION_TO_NAME[int(gym_action_id)]]
