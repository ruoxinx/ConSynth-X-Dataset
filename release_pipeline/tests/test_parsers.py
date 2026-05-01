"""Round-trip and invariant tests for VOC and CS10k converters.

Run from ConSynth-X/ root:
    python -m release_pipeline.tests.test_parsers
"""
import sys
import json

sys.path.insert(0, ".")

from release_pipeline.convert.voc_to_objects import (
    parse_voc_xml, normalise_objects, denormalise_for_round_trip,
)
from release_pipeline.convert.cs10k_to_objects import (
    from_format_a, from_format_b, detect_format, CS10K_CLASS_TO_ID,
)
from release_pipeline.convert.release_schema import (
    SCHEMAS, validate_table, compute_quality_alert,
)


# ---------- VOC ----------

VOC_XML = """<?xml version="1.0"?>
<annotation>
  <size><width>1000</width><height>800</height></size>
  <object><name>person</name><bndbox><xmin>100</xmin><ymin>200</ymin><xmax>400</xmax><ymax>600</ymax></bndbox></object>
  <object><name>excavator</name><bndbox><xmin>50.5</xmin><ymin>50.5</ymin><xmax>150.5</xmax><ymax>250.5</ymax></bndbox></object>
</annotation>
"""


def test_voc_round_trip():
    parsed = parse_voc_xml(VOC_XML)
    assert parsed.raw_size_in_xml == (1000, 800)
    assert len(parsed.objects) == 2
    cls_map = {"person": 0, "excavator": 1}
    objs, warns = normalise_objects(parsed, 1000, 800, cls_map)
    assert warns == []
    assert objs[0]["class_name"] == "person"
    assert objs[0]["class_id"] == 0
    # Round-trip first object
    rt = denormalise_for_round_trip(objs[0], 1000, 800)
    assert rt == (100.0, 200.0, 400.0, 600.0), rt
    rt2 = denormalise_for_round_trip(objs[1], 1000, 800)
    assert rt2 == (50.5, 50.5, 150.5, 250.5), rt2
    print("test_voc_round_trip OK")


def test_voc_size_mismatch_raises():
    parsed = parse_voc_xml(VOC_XML)
    try:
        normalise_objects(parsed, 2000, 1600, {"person": 0, "excavator": 1})
    except ValueError as e:
        assert "disagrees with image" in str(e)
        print("test_voc_size_mismatch_raises OK")
        return
    raise AssertionError("expected ValueError on VOC <size> mismatch")


def test_voc_unknown_class_raises():
    parsed = parse_voc_xml(VOC_XML)
    try:
        normalise_objects(parsed, 1000, 800, {"person": 0})  # missing 'excavator'
    except ValueError as e:
        assert "excavator" in str(e)
        print("test_voc_unknown_class_raises OK")
        return
    raise AssertionError("expected ValueError")


def test_voc_out_of_bounds_raises():
    bad = """<?xml version="1.0"?>
<annotation>
  <size><width>100</width><height>100</height></size>
  <object><name>person</name><bndbox><xmin>50</xmin><ymin>50</ymin><xmax>200</xmax><ymax>120</ymax></bndbox></object>
</annotation>"""
    parsed = parse_voc_xml(bad)
    try:
        normalise_objects(parsed, 100, 100, {"person": 0})
    except ValueError as e:
        assert "out of image bounds" in str(e)
        print("test_voc_out_of_bounds_raises OK")
        return
    raise AssertionError("expected ValueError")


def test_voc_degenerate_skipped_with_warning():
    bad = """<?xml version="1.0"?>
<annotation>
  <size><width>1000</width><height>1000</height></size>
  <object><name>person</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object>
  <object><name>slogan</name><bndbox><xmin>525</xmin><ymin>839</ymin><xmax>525</xmax><ymax>841</ymax></bndbox></object>
</annotation>"""
    parsed = parse_voc_xml(bad)
    assert len(parsed.objects) == 2
    objs, warns = normalise_objects(parsed, 1000, 1000, {"person": 0, "slogan": 1})
    assert len(objs) == 1, objs
    assert objs[0]["class_name"] == "person"
    assert any(w["kind"] == "degenerate_voc_bbox" and w["class"] == "slogan" for w in warns), warns
    print("test_voc_degenerate_skipped_with_warning OK")


# ---------- CS10k ----------

CS10K_FORMAT_A = {
    "image_id": "0000005",
    "image_caption": "An excavator scene.",
    "illumination": "normal lighting",
    "camera_distance": "mid distance",
    "view": "elevation view",
    "quality_of_info": "rich info",
    "rule_1_violation": {"bounding_box": [[0.14, 0.59, 0.19, 0.7]], "reason": "Person on the left not using PPE."},
    "rule_2_violation": None,
    "rule_3_violation": None,
    "rule_4_violation": None,
    "excavator": [[0.23, 0.29, 0.82, 0.83]],
    "rebar": [],
    "worker_with_white_hard_hat": [],
    "ref_id": "0000005",
}


def test_cs10k_format_a():
    canon = from_format_a(CS10K_FORMAT_A)
    assert len(canon.objects) == 1
    assert canon.objects[0]["class_name"] == "excavator"
    assert canon.objects[0]["class_id"] == CS10K_CLASS_TO_ID["excavator"]
    assert canon.objects[0]["bbox"] == [0.23, 0.29, 0.82, 0.83]
    assert len(canon.rule_violations) == 1
    rv = canon.rule_violations[0]
    assert rv["rule_id"] == 1
    assert rv["bbox"] == [0.14, 0.59, 0.19, 0.7]
    assert "PPE" in rv["reason"]
    assert canon.image_attributes["caption"].startswith("An excavator")
    print("test_cs10k_format_a OK")


CS10K_FORMAT_B_JSON = json.dumps({
    "image_id": "0000015",
    "bbox_format": "xyxy",
    "excavator": [],
    "rebar": [[0.272344, 0.489888, 0.692243, 0.752809]],
    "worker_with_white_hard_hat": [],
    "rule_1_violation": None,
    "rule_2_violation": {"bounding_box": [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]], "reason": "rule 2"},
    "rule_3_violation": None,
    "rule_4_violation": None,
})


def test_cs10k_format_b():
    canon = from_format_b(CS10K_FORMAT_B_JSON)
    assert len(canon.objects) == 1
    assert canon.objects[0]["class_name"] == "rebar"
    assert len(canon.rule_violations) == 2
    assert all(rv["rule_id"] == 2 for rv in canon.rule_violations)
    print("test_cs10k_format_b OK")


def test_cs10k_format_b_bad_bbox_format_raises():
    bad = json.dumps({"bbox_format": "xywh", "excavator": [], "rebar": [], "worker_with_white_hard_hat": []})
    try:
        from_format_b(bad)
    except ValueError as e:
        assert "bbox_format" in str(e)
        print("test_cs10k_format_b_bad_bbox_format_raises OK")
        return
    raise AssertionError("expected ValueError")


def test_cs10k_detect_format():
    assert detect_format(CS10K_FORMAT_A) == "A"
    assert detect_format({"image_id": "x", "annotation": CS10K_FORMAT_B_JSON}) == "B"
    try:
        detect_format({"image_id": "x"})
    except ValueError:
        print("test_cs10k_detect_format OK")
        return
    raise AssertionError("expected ValueError")


def test_cs10k_rule_with_empty_bbox_preserved():
    """Rule fires with reason text but no bbox: must keep one entry with bbox=None."""
    row = dict(CS10K_FORMAT_A)
    row["rule_2_violation"] = {"bounding_box": [], "reason": "fired but empty"}
    canon = from_format_a(row)
    rule2 = [rv for rv in canon.rule_violations if rv["rule_id"] == 2]
    assert len(rule2) == 1, rule2
    assert rule2[0]["bbox"] is None
    assert rule2[0]["reason"] == "fired but empty"
    print("test_cs10k_rule_with_empty_bbox_preserved OK")


def test_cs10k_rule_multi_bbox_emits_multi_entries():
    row = dict(CS10K_FORMAT_A)
    row["rule_3_violation"] = {"bounding_box": [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]],
                                "reason": "two violators"}
    canon = from_format_a(row)
    rule3 = [rv for rv in canon.rule_violations if rv["rule_id"] == 3]
    assert len(rule3) == 2
    assert all(rv["reason"] == "two violators" for rv in rule3)
    print("test_cs10k_rule_multi_bbox_emits_multi_entries OK")


# ---------- quality alert ----------

def test_quality_alert():
    assert compute_quality_alert(None) is None
    assert compute_quality_alert(0.74) is True
    assert compute_quality_alert(0.75) is False
    assert compute_quality_alert(0.90) is False
    print("test_quality_alert OK")


def main():
    test_voc_round_trip()
    test_voc_size_mismatch_raises()
    test_voc_unknown_class_raises()
    test_voc_out_of_bounds_raises()
    test_voc_degenerate_skipped_with_warning()
    test_cs10k_format_a()
    test_cs10k_format_b()
    test_cs10k_format_b_bad_bbox_format_raises()
    test_cs10k_detect_format()
    test_cs10k_rule_with_empty_bbox_preserved()
    test_cs10k_rule_multi_bbox_emits_multi_entries()
    test_quality_alert()
    print("\nALL TESTS PASS")


if __name__ == "__main__":
    main()
