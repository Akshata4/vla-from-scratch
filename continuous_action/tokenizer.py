"""
Instruction tokenizer for the continuous-action VLA variant.

The big difference from grid_world/tokenizer.py: there are NO action tokens
in this vocabulary at all. The discrete version (grid_world) extends the
vocabulary with <UP>/<DOWN>/<LEFT>/<RIGHT> and predicts one of them via
softmax -- that's the whole RT-2-style trick. This variant throws that idea
out entirely: the model just outputs two raw numbers (dx, dy) directly, so
there's nothing for an "action vocabulary" to do here.
"""

PAD = "<PAD>"
WORDS = ["go", "to", "the", "red", "green", "blue", "yellow"]

VOCAB = [PAD] + WORDS
VOCAB_SIZE = len(VOCAB)
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
PAD_ID = TOKEN_TO_ID[PAD]

MAX_TEXT_LEN = 4  # "go to the <color>"


def encode_instruction(instruction):
    ids = [TOKEN_TO_ID[w] for w in instruction.split()]
    ids += [PAD_ID] * (MAX_TEXT_LEN - len(ids))
    return ids


# The continuous training TARGET each named action maps to. These numbers
# are not arbitrary -- they match env.py's own ACTION_DELTA exactly, since
# both describe the same (row, column) movement. This is the label the
# model gets supervised against, just expressed as a 2D vector instead of
# a vocabulary index.
ACTION_VECTOR = {
    "up": (-1.0, 0.0),
    "down": (1.0, 0.0),
    "left": (0.0, -1.0),
    "right": (0.0, 1.0),
}


def encode_action(action_name):
    return ACTION_VECTOR[action_name]


def vector_to_action(dx, dy):
    """Inverse of encode_action(): snap a PREDICTED continuous (dx, dy) back
    to one of the 4 moves env.step() actually understands, so we can still
    execute it in the same discrete-grid environment.

    This is the one place a "continuous" policy still has to touch a
    discrete action space, because our environment only understands 4
    moves -- a real continuous-control robot wouldn't need this step, its
    actuators would just take the raw (dx, dy) directly. Uses the same
    larger-axis-wins tie-break as env.py's own expert_action(), so this
    function is the mirror image of that one.
    """
    if abs(dx) >= abs(dy):
        return "down" if dx > 0 else "up"
    return "right" if dy > 0 else "left"
