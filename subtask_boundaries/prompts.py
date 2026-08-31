"""Prompts used by the coarse and dense temporal-boundary passes."""

# Kept verbatim for A/B comparison against COARSE_BOUNDARY_PROMPT below. This
# version under-segments badly (3-7 boundaries on 300-700 frame episodes, vs.
# 15-25 for AWE) because every granularity rule pushes one way -- eight "Do NOT"
# clauses, "prefer fewer", "if unsure DO NOT", "prioritize correctness over
# producing many" -- with nothing anchoring how many boundaries a good answer
# has. It also never mentions the gripper, the single most legible visual cue.
COARSE_BOUNDARY_PROMPT_V1 = """
Reconstruct the sequence of COMPLETED manipulation events in this robot trajectory
from the timestamped contact sheets.

Each image has its ORIGINAL trajectory frame index visibly printed on it.

A boundary marks the FIRST visible frame of a new completed manipulation event.

Count only atomic manipulation events that produce a persistent world-state change.

STRICT GRANULARITY RULES:
- Segment completed manipulation events, NOT every visible movement.
- Prefer fewer well-supported boundaries over speculative boundaries.
- If unsure whether a candidate is a true state change or merely intermediate motion,
  DO NOT create a boundary.

Return structured output only:
{"boundary_indices": [...]}
"""

# Kept verbatim for A/B comparison. V2 fixed V1's under-segmentation (5-7 -> 13-18
# boundaries on Kitchen) but is unstable: on 2 of 5 demos it degenerated into
# labelling nearly every sampled frame (56 and 61 boundaries, median gap exactly
# equal to the sampling stride). Cause: three one-way density pushes -- an explicit
# per-episode count target, "3-6x more boundaries than a first instinct", and "if a
# candidate is plausible, include it" -- with no rule for what disqualifies a frame.
COARSE_BOUNDARY_PROMPT_V2 = """
You are labelling SUBGOAL frames in a robot manipulation trajectory, from
timestamped contact sheets. Each image has its ORIGINAL trajectory frame index
printed on it.

{task_context}

WHAT A SUBGOAL IS
A subgoal is a frame where the gripper HAS ARRIVED at a pose that matters for the
task. Label the frame where the arrival is first clearly visible -- the moment of
the event itself, not the motion leading into it and not a later frame once the
next motion has begun.

Label a subgoal at each of these, whenever they occur:
- the gripper has arrived at an object and is positioned to grasp it (pre-grasp),
- the gripper has closed on an object (grasp),
- a grasped object has been lifted clear of its support,
- the gripper carrying an object has arrived over/at its destination (pre-place),
- the gripper has opened and released the object (release),
- a drawer, door, lid, or handle has been reached, and separately once it has been
  fully opened or closed,
- a tool has made contact with its target, and separately once its effect is done,
- contents have finished being poured or transferred,
- the arm has arrived at a distinctly new region of the workspace to begin the next
  part of the task.

GRIPPER CUE
The gripper fingers are the most reliable signal in these images. Every frame where
the fingers visibly change between open and closed is a subgoal. Check the fingers
in every sampled frame before deciding, and never skip a visible open/close.

HOW MANY
{density_hint}
A correct answer is usually 3-6x more boundaries than a first instinct suggests:
these episodes contain multiple pick-and-place cycles, and each cycle contributes
several subgoals (pre-grasp, grasp, lift, pre-place, release), not one.
Under-labelling is the most common failure. If a candidate is plausible, include it
-- a slightly redundant subgoal is far less harmful than a missing one. Only omit a
frame when nothing about the gripper or the objects has changed since the previous
boundary you kept.

WHAT IS NOT A SUBGOAL
- Pure transport with the object held and nothing arriving or changing.
- Two frames describing the same arrival; keep only the first.

PRECISION
Boundaries must be exact. When an event is visible across several sampled frames,
choose the FIRST frame in which it is unambiguously true. Use only frame indices
visibly shown in the contact sheets.

Return structured output only:
{{"boundary_indices": [...]}}
"""

# A "check the fingers in EVERY image / never skip an open-close" instruction is
# what makes this degenerate: V2 and the C2 variant both carried it and both
# collapsed into labelling nearly every sampled frame on 3 of 5 Kitchen demos
# (56-61 boundaries, median gap = the sampling stride). At agentview resolution
# the fingers are a few pixels wide, so a rule that hangs on them makes the model
# hedge by emitting everything. Do not reintroduce it without switching the
# refinement pass to the wrist camera, where finger state is actually legible.
# V3 (live). Same text for every task -- the only per-task content is the
# instruction interpolated into {task_context}. Density is controlled by what
# QUALIFIES as an event, never by a target count: V2 showed that any number in the
# prompt is either ignored or over-shot, and a count would have to be retuned per
# task anyway. The closed event list plus the adjacency rule are what keep a
# 600-frame episode from collapsing into "every sampled frame is a subgoal".
COARSE_BOUNDARY_PROMPT = """
You are labelling SUBGOAL frames in a robot manipulation trajectory, shown as
timestamped contact sheets. Each image has its ORIGINAL trajectory frame index
printed on it.

{task_context}

WHAT COUNTS AS A SUBGOAL
A subgoal is a frame in which one DISCRETE, VISIBLE event completes. An event is
one of the following, and nothing else:
  (a) the gripper fingers close onto an object,
  (b) the gripper fingers open and release an object,
  (c) a held object first loses contact with the surface or support it rested on,
  (d) a held object first makes contact with the surface, container, or fixture it
      is being placed into or onto,
  (e) an articulated part (drawer, door, lid, handle, button, switch) first reaches
      its fully opened, fully closed, or fully pressed position,
  (f) the gripper, empty or loaded, first makes contact with the object or fixture
      it is about to act on.

Every boundary you return must be one of (a)-(f). Before emitting a frame, name
which letter it is. If no letter applies, that frame is not a subgoal.

WHAT IS NOT A SUBGOAL
- Moving toward something, however far the arm travels, until contact in (f).
- Carrying an object between two places.
- Lifting or lowering that is already covered by the (c) or (d) frame.
- Retreating, hesitating, regrasping, or small adjustments.
- A frame that describes the same event as a boundary you already emitted -- one
  event produces exactly ONE boundary, the earliest frame where it is true.
- A frame chosen because a lot of time has passed since the last boundary. Elapsed
  time is never a reason; only (a)-(f) are.

DENSITY
Both failure modes are real. Do not compress a whole pick-and-place into one
boundary: closing on the object (a), lifting it clear (c), touching down at the
destination (d), and releasing it (b) are four separate events and each gets its
own frame. But equally, {stride_note}.
Two boundaries in adjacent or nearly adjacent images therefore require two
genuinely different events from the list -- if you cannot name a different letter
for each, emit only the first. Watch the gripper
fingers in every image: a visible change between open and closed is always an
event, and is the single most reliable cue in these images.

PRECISION
When an event is visible across several sampled images, choose the FIRST image in
which it is unambiguously true. Use only frame indices visibly printed in the
contact sheets.

Return structured output only:
{{"boundary_indices": [...]}}
"""


_NO_INSTRUCTION_CONTEXT = (
    "The episode is a multi-step tabletop manipulation task performed by a robot arm."
)


def _task_context(instruction: str | None) -> str:
    if instruction and instruction.strip():
        return (
            "TASK BEING PERFORMED\n"
            f"{instruction.strip()}\n"
            "Use this to decide which arrivals and state changes are task-relevant."
        )
    return f"TASK BEING PERFORMED\n{_NO_INSTRUCTION_CONTEXT}"


def _stride_note(sample_every_n_frames: int | None) -> str:
    """Describe the temporal resolution WITHOUT implying a boundary count.

    The model needs to know that consecutive images are close together in time --
    otherwise it treats every image as a fresh scene and labels most of them. It
    must not be told how many boundaries to produce: V2 did, and either overshot
    the number badly or ignored it, and any count would need retuning per task.
    """
    if sample_every_n_frames and sample_every_n_frames > 0:
        return (
            f"consecutive images are only {sample_every_n_frames} trajectory frames "
            "apart and usually show the same ongoing motion"
        )
    return "consecutive images are close together in time and usually show the same ongoing motion"


def coarse_prompt(
    instruction: str | None = None,
    *,
    sample_every_n_frames: int | None = None,
) -> str:
    """Return the coarse prompt with task context and the sampling resolution.

    Task-agnostic by construction: `instruction` is the only per-task content, and
    everything else is fixed text. V1 interpolated neither and -- because it never
    called .format() -- shipped a literal '{instruction}' placeholder to the model.
    """
    return COARSE_BOUNDARY_PROMPT.format(
        task_context=_task_context(instruction),
        stride_note=_stride_note(sample_every_n_frames),
    )


def refinement_prompt(coarse_index: int, instruction: str | None = None) -> str:
    """Return the single-boundary dense-refinement prompt."""
    context = ""
    if instruction and instruction.strip():
        context = f"\nTask being performed: {instruction.strip()}\n"

    return f"""A coarse pass placed a subgoal near frame {coarse_index}, but that estimate
comes from frames sampled far apart and is often off by tens of frames.
{context}
These images show a DENSE window around that estimate. Each image has its ORIGINAL
trajectory frame index printed on it.

Find the exact frame where the event happens -- the first frame in which the gripper
has ARRIVED at its pose, or in which the fingers have visibly finished opening or
closing, or in which the object has visibly changed state.

Watch the gripper fingers across consecutive frames: the frame where they first reach
their new open/closed position is the answer. Do not default to {coarse_index}; pick it
only if it is genuinely the best frame in this window. Prefer the earliest frame in
which the new state is unambiguously true.

Use only an original frame index visibly shown in these contact sheets.

Return structured output only:
{{"boundary_index": <integer>}}
"""
