from livephoto2lrhr.stages.color_match import ColorMatchStage


def test_color_match_stage_is_explicitly_not_implemented():
    stage = ColorMatchStage(enabled=False)

    assert stage.enabled is False
    assert stage.describe() == "color_match stage is reserved for phase 3"
