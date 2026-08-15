"""Prompts used by the coarse and dense temporal-boundary passes."""

COARSE_BOUNDARY_PROMPT = """You are identifying temporal transition points in a robot manipulation trajectory.

The images are sampled from the trajectory and arranged into contact sheets.
Each image has its ORIGINAL trajectory frame index visibly printed on it.

Identify frame indices where a meaningful manipulation state transitions into a new task-relevant state.

A transition can occur when:
- an object becomes securely grasped
- an object is released
- an object reaches its intended destination
- an object is inserted or removed
- a drawer, door, container, or lid becomes opened or closed
- another persistent task-relevant world state changes

Do NOT create transition points for:
- reaching or approaching
- generic robot arm motion
- grasp adjustment
- small repositioning
- hesitation
- camera motion
- retreat

A boundary represents the FIRST frame of the new task state.
Use only frame indices visibly shown in the contact sheets.
Prioritize temporally correct boundaries.

Return structured output only:
{"boundary_indices": [...]}
"""


def coarse_prompt(instruction: str | None = None) -> str:
    """Return the coarse prompt, optionally augmented with task context."""
    if instruction and instruction.strip():
        return f"{COARSE_BOUNDARY_PROMPT}\nEpisode instruction: {instruction.strip()}\n"
    return COARSE_BOUNDARY_PROMPT


def refinement_prompt(coarse_index: int, instruction: str | None = None) -> str:
    """Return the single-boundary dense-refinement prompt."""
    text = f"""The previous coarse analysis determined that a task transition occurs near frame {coarse_index}.

Examine this dense sequence. Each image shows its ORIGINAL trajectory frame index.
Find the SINGLE frame that best represents the first frame of the new task-relevant state.
Use only an original frame index visibly shown in these contact sheets.

Return structured output only:
{{"boundary_index": <integer>}}
"""
    if instruction and instruction.strip():
        text += f"\nEpisode instruction: {instruction.strip()}\n"
    return text

