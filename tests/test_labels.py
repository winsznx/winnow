from pathlib import Path

import pytest

from winnow.labels import GateFailure, build_labels, label_clips, parse_timeline

FCPXML = """<?xml version="1.0"?>
<fcpxml version="1.10"><resources>
<asset id="r1" name="A001"><media-rep kind="original-media" src="file:///Volumes/shoot/A001.MOV"/></asset>
<asset id="r2" name="A002"><media-rep kind="original-media" src="file:///Volumes/shoot/A002.MOV"/></asset>
</resources><library><event><project><sequence><spine>
<asset-clip ref="r1" offset="0s" start="100/25s" duration="50/25s" name="A001"/>
<asset-clip ref="r1" offset="2s" start="10s" duration="3s" name="A001"/>
</spine></sequence></project></event></library></fcpxml>"""

XMEML = """<?xml version="1.0"?>
<xmeml version="4"><sequence><media><video><track>
<clipitem id="c1"><in>25</in><out>75</out><rate><timebase>25</timebase></rate>
<file id="f1"><name>B001.mp4</name><pathurl>file://localhost/Volumes/s/B001.mp4</pathurl></file></clipitem>
<clipitem id="c2"><in>0</in><out>50</out><rate><timebase>25</timebase></rate><file id="f1"/></clipitem>
</track></video></media></sequence></xmeml>"""

EDL = """TITLE: cut
FCM: NON-DROP FRAME

001  C0001    V     C        00:00:01:00 00:00:03:00 01:00:00:00 01:00:02:00
* FROM CLIP NAME: C0001.MP4
002  C0003    V     C        00:00:10:12 00:00:11:00 01:00:02:00 01:00:02:12
003  BL       V     C        00:00:00:00 00:00:01:00 01:00:02:12 01:00:03:12
"""


def make_clips(tmp_path: Path, names: list[str]) -> list[Path]:
    paths = [tmp_path / n for n in names]
    for p in paths:
        p.touch()
    return paths


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_fcpxml_labels_and_ranges(tmp_path):
    uses = parse_timeline(write(tmp_path, "t.fcpxml", FCPXML))
    result = build_labels(uses, make_clips(tmp_path, ["A001.MOV", "A002.MOV"]))
    assert result.labels == {"A001.MOV": 1, "A002.MOV": 0}
    assert result.ranges["A001.MOV"] == [(4.0, 6.0), (10.0, 13.0)]
    assert result.unmatched_sources == []


def test_xmeml_resolves_file_reference_by_id(tmp_path):
    uses = parse_timeline(write(tmp_path, "t.xml", XMEML))
    assert [u.name for u in uses] == ["B001.mp4", "B001.mp4"]
    result = build_labels(uses, make_clips(tmp_path, ["B001.mp4", "B002.mp4"]))
    assert result.labels == {"B001.mp4": 1, "B002.mp4": 0}
    assert result.ranges["B001.mp4"] == [(1.0, 3.0), (0.0, 2.0)]


def test_edl_matches_by_clip_name_then_reel_and_skips_black(tmp_path):
    uses = parse_timeline(write(tmp_path, "t.edl", EDL), fps=24)
    assert [u.name for u in uses] == ["C0001.MP4", "C0003"]
    result = build_labels(uses, make_clips(tmp_path, ["C0001.MP4", "C0002.MP4", "C0003.MP4"]))
    assert result.labels == {"C0001.MP4": 1, "C0002.MP4": 0, "C0003.MP4": 1}
    assert result.ranges["C0003.MP4"] == [(10.5, 11.0)]


def test_unmatched_report_counts_sources(tmp_path):
    uses = parse_timeline(write(tmp_path, "t.fcpxml", FCPXML))
    result = build_labels(uses, make_clips(tmp_path, ["A002.MOV"]))
    assert result.unmatched_sources == ["A001.MOV"]
    assert result.unmatched_fraction == 1.0
    assert "unmatched: A001.MOV" in result.report()


def test_gate_two_fails_over_ten_percent(tmp_path):
    t = write(tmp_path, "t.edl", EDL)
    clips = tmp_path / "clips"
    clips.mkdir()
    make_clips(clips, ["C0001.MP4"])
    with pytest.raises(GateFailure):
        label_clips(clips, t)
