# A minimal vocabulary, built the way RT-2/OpenVLA build theirs: take a
# language vocabulary and append action tokens to it. <BOA> ("beginning of
# action") is the query position the model uses to predict the next token,
# which for us is always one of the action tokens.
PAD, BOA = "<PAD>", "<BOA>"
WORDS = ["go", "to", "the", "red", "green", "blue", "yellow"]
ACTION_NAMES = ["up", "down", "left", "right"]
ACTION_TOKENS = [f"<{name.upper()}>" for name in ACTION_NAMES]

VOCAB = [PAD, BOA] + WORDS + ACTION_TOKENS
VOCAB_SIZE = len(VOCAB)
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}

PAD_ID = TOKEN_TO_ID[PAD]
BOA_ID = TOKEN_TO_ID[BOA]
ACTION_IDS = [TOKEN_TO_ID[t] for t in ACTION_TOKENS]
ACTION_NAME_TO_ID = {name: TOKEN_TO_ID[tok] for name, tok in zip(ACTION_NAMES, ACTION_TOKENS)}
ID_TO_ACTION_NAME = {v: k for k, v in ACTION_NAME_TO_ID.items()}

MAX_TEXT_LEN = 4  # "go to the <color>"


def encode_instruction(instruction):
    ids = [TOKEN_TO_ID[w] for w in instruction.split()]
    ids += [PAD_ID] * (MAX_TEXT_LEN - len(ids))
    return ids


def encode_action(action_name):
    return ACTION_NAME_TO_ID[action_name]
