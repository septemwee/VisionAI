from services.viewport_service import select_viewport


def test_viewport_uses_inner_image_not_outer_container():
    candidates = [("ImageView", (10, 20, 800, 600)),
                  ("ImageViewport", (20, 40, 700, 500)),
                  ("Toolbar", (0, 0, 900, 30))]
    assert select_viewport(candidates, (0, 0, 1000, 800)) == (20, 40, 700, 500)


def test_viewport_rejects_unknown_and_ambiguous_children():
    bounds = (0, 0, 1000, 800)
    assert select_viewport([("Panel", bounds)], bounds) is None
    assert select_viewport([("ImageView", (0, 0, 300, 300)),
                            ("ImageView", (400, 0, 300, 300))], bounds) is None
