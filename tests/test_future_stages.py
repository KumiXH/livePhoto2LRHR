from livephoto2lrhr.config import ColorMatchConfig


def test_color_match_stage_is_disabled_by_default():
    config = ColorMatchConfig()

    assert config.enabled is False
