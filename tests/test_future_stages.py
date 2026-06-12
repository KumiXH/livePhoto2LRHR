from livephoto2lrhr.stages.align import AlignStage
from livephoto2lrhr.stages.color_match import ColorMatchStage


def test_align_stage_is_explicitly_not_implemented():
    stage = AlignStage(enabled=False)

    assert stage.enabled is False
    assert stage.describe() == "align stage is reserved for phase 2"


def test_color_match_stage_is_explicitly_not_implemented():
    stage = ColorMatchStage(enabled=False)

    assert stage.enabled is False
    assert stage.describe() == "color_match stage is reserved for phase 3"
