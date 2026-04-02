"""Task definitions for v0.5 evaluation.

Provides a curated set of object rearrangement tasks across multiple
AI2-THOR kitchen scenes. Tasks range from easy (object visible, close
to target) to hard (object inside a closed container, far from target).
"""

from __future__ import annotations

from src.thor.schemas import TaskDefinition


def get_rearrangement_tasks() -> list[TaskDefinition]:
    """Return a set of object rearrangement tasks for v0.5 evaluation.

    Each task asks the agent to find an object and move it to a target
    receptacle. Difficulty varies by navigation distance and object type.
    Tasks span FloorPlan1 through FloorPlan5 and cover six object types
    and five receptacle types.

    Returns:
        A list of 15 TaskDefinition instances for evaluation.
    """
    return [
        # --- Easy: object likely visible, target nearby ---
        TaskDefinition(
            task_id="rearrange_001",
            scene_name="FloorPlan1",
            target_object_type="Mug",
            target_receptacle_type="DiningTable",
            max_steps=20,
            description="Find the mug and place it on the dining table.",
        ),
        TaskDefinition(
            task_id="rearrange_002",
            scene_name="FloorPlan1",
            target_object_type="Apple",
            target_receptacle_type="CounterTop",
            max_steps=20,
            description="Find an apple and place it on the countertop.",
        ),
        TaskDefinition(
            task_id="rearrange_003",
            scene_name="FloorPlan2",
            target_object_type="Bowl",
            target_receptacle_type="DiningTable",
            max_steps=20,
            description="Find a bowl and place it on the dining table.",
        ),
        TaskDefinition(
            task_id="rearrange_004",
            scene_name="FloorPlan2",
            target_object_type="Book",
            target_receptacle_type="Shelf",
            max_steps=25,
            description="Find a book and place it on a shelf.",
        ),
        # --- Medium: may need to navigate or open something ---
        TaskDefinition(
            task_id="rearrange_005",
            scene_name="FloorPlan3",
            target_object_type="Plate",
            target_receptacle_type="CounterTop",
            max_steps=25,
            description="Find a plate and place it on the countertop.",
        ),
        TaskDefinition(
            task_id="rearrange_006",
            scene_name="FloorPlan3",
            target_object_type="Mug",
            target_receptacle_type="SinkBasin",
            max_steps=25,
            description="Find the mug and place it in the sink basin.",
        ),
        TaskDefinition(
            task_id="rearrange_007",
            scene_name="FloorPlan1",
            target_object_type="Knife",
            target_receptacle_type="CounterTop",
            max_steps=25,
            description="Find a knife and place it on the countertop.",
        ),
        TaskDefinition(
            task_id="rearrange_008",
            scene_name="FloorPlan4",
            target_object_type="Apple",
            target_receptacle_type="DiningTable",
            max_steps=25,
            description="Find an apple and place it on the dining table.",
        ),
        TaskDefinition(
            task_id="rearrange_009",
            scene_name="FloorPlan4",
            target_object_type="Bowl",
            target_receptacle_type="Shelf",
            max_steps=25,
            description="Find a bowl and place it on a shelf.",
        ),
        TaskDefinition(
            task_id="rearrange_010",
            scene_name="FloorPlan5",
            target_object_type="Mug",
            target_receptacle_type="CounterTop",
            max_steps=25,
            description="Find the mug and place it on the countertop.",
        ),
        # --- Hard: object likely in a cabinet or fridge, longer navigation ---
        TaskDefinition(
            task_id="rearrange_011",
            scene_name="FloorPlan5",
            target_object_type="Plate",
            target_receptacle_type="DiningTable",
            max_steps=30,
            description=(
                "Find a plate (it may be inside a cabinet) and place it "
                "on the dining table."
            ),
        ),
        TaskDefinition(
            task_id="rearrange_012",
            scene_name="FloorPlan3",
            target_object_type="Apple",
            target_receptacle_type="Fridge",
            max_steps=30,
            description=(
                "Find an apple and place it inside the fridge. You may "
                "need to open the fridge first."
            ),
        ),
        TaskDefinition(
            task_id="rearrange_013",
            scene_name="FloorPlan2",
            target_object_type="Knife",
            target_receptacle_type="SinkBasin",
            max_steps=30,
            description=(
                "Find a knife and place it in the sink basin. The knife "
                "may be in a drawer."
            ),
        ),
        TaskDefinition(
            task_id="rearrange_014",
            scene_name="FloorPlan4",
            target_object_type="Book",
            target_receptacle_type="Shelf",
            max_steps=30,
            description=(
                "Find a book and place it on a shelf. You may need to "
                "navigate through multiple rooms."
            ),
        ),
        TaskDefinition(
            task_id="rearrange_015",
            scene_name="FloorPlan5",
            target_object_type="Bowl",
            target_receptacle_type="Fridge",
            max_steps=30,
            description=(
                "Find a bowl, open the fridge, and place the bowl inside. "
                "This requires multiple interaction steps."
            ),
        ),
    ]
