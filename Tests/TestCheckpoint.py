import pytest

from supermimic.Checkpoint import check_checkpoint, map_metadata, write_run_metadata


def metadata():
    return map_metadata({"profile": "flat", "cell_size": .05, "bounds": [-2, 2, -2, 2], "boxes": []}, 32)


def test_resume_rejects_upstream_and_evaluation_only_checkpoints():
    with pytest.raises(ValueError, match="heightmap"):
        check_checkpoint({"pi": {}, "wm": {}}, metadata(), resume=True)
    ck = {"heightmap": metadata(), "pi": {}, "wm": {}}
    check_checkpoint(ck, metadata(), resume=False)
    with pytest.raises(ValueError, match="last.pt"):
        check_checkpoint(ck, metadata(), resume=True)


def test_changed_terrain_cannot_silently_reuse_checkpoint():
    ck = {"heightmap": metadata(), "pi": {}}
    changed = map_metadata({"profile": "steps", "cell_size": .05, "bounds": [-2, 2, -2, 2],
                            "boxes": [[1, 0, .5, 1, .04]]}, 32)
    with pytest.raises(ValueError, match="configuration"):
        check_checkpoint(ck, changed)


def test_resume_accepts_matching_full_checkpoint():
    ck = {k: {} for k in ("pi", "wm", "critic", "opt_pi", "opt_wm", "opt_v", "args")}
    ck.update(heightmap=metadata(), iteration=42, wall_time=1., sim_steps=100)
    check_checkpoint(ck, metadata(), resume=True)


def test_failed_resume_does_not_overwrite_original_run_metadata(tmp_path):
    write_run_metadata(tmp_path, {"terrain": "flat"}, metadata(), resume=False)
    before = {name: (tmp_path / name).read_bytes() for name in ("Heightmap.json", "args.json")}
    changed = map_metadata({"profile": "steps"}, 32)
    with pytest.raises(ValueError, match="configuration"):
        write_run_metadata(tmp_path, {"terrain": "steps"}, changed, resume=True)
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    with pytest.raises(FileExistsError):
        write_run_metadata(tmp_path, {}, metadata(), resume=False)
    write_run_metadata(tmp_path, {"resume": "last.pt"}, metadata(), resume=True)
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    assert (tmp_path / "Resume.json").exists()
