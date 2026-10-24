import pytest
import torch

from supermimic.Training import (
    build_reset_candidates,
    terrain_reset_mask,
    update_bin_returns,
    warmup_steps_needed,
)


def test_duplicate_bin_returns_are_averaged_before_one_ema_update():
    values = torch.tensor([3.0, 10.0, 7.0])
    update_bin_returns(values, torch.tensor([1, 1]), torch.tensor([4.0, 8.0]))
    torch.testing.assert_close(values, torch.tensor([3.0, 9.2, 7.0]))


def test_duplicate_bin_update_does_not_depend_on_episode_order():
    first = torch.tensor([10.0, 20.0, 30.0])
    second = first.clone()
    update_bin_returns(first, torch.tensor([2, 0, 2]), torch.tensor([4.0, 2.0, 8.0]))
    update_bin_returns(second, torch.tensor([2, 2, 0]), torch.tensor([8.0, 4.0, 2.0]))
    torch.testing.assert_close(first, torch.tensor([8.4, 20.0, 25.2]))
    torch.testing.assert_close(first, second)


def test_unique_bins_retain_existing_ema_behavior_and_empty_input_is_noop():
    values = torch.tensor([10.0, 20.0, 30.0])
    update_bin_returns(values, torch.tensor([0, 2]), torch.tensor([2.0, 10.0]))
    torch.testing.assert_close(values, torch.tensor([8.4, 20.0, 26.0]))
    update_bin_returns(values, torch.empty(0, dtype=torch.long), torch.empty(0))
    torch.testing.assert_close(values, torch.tensor([8.4, 20.0, 26.0]))


@pytest.mark.parametrize(
    "size,capacity,window,warmup,expected",
    [(0, 1000, 8, 64, 64), (2, 1000, 8, 64, 62), (64, 1000, 8, 64, 0),
     (0, 1000, 8, 0, 8), (5, 16, 8, 64, 11), (16, 16, 8, 64, 0)],
)
def test_resume_refills_missing_replay_steps_before_window_sampling(size, capacity, window, warmup, expected):
    assert warmup_steps_needed(size, capacity, window, warmup) == expected


@pytest.mark.parametrize("capacity,window", [(7, 8), (0, 8)])
def test_replay_capacity_that_cannot_hold_one_model_window_is_rejected(capacity, window):
    with pytest.raises(ValueError, match="capacity"):
        warmup_steps_needed(0, capacity, window, 64)


def test_reset_mask_rejects_body_reach_near_box_without_changing_reference_height():
    roots = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    bodies = torch.tensor([[[0.0, 0.0], [0.4, 0.0]],
                           [[2.0, 0.0], [2.4, 0.0]],
                           [[0.0, 2.0], [0.4, 2.0]]])
    boxes = ((1.0, 0.0, 0.5, 0.5, 0.03),)
    # Body reach .4 plus safety margin .4 intersects the first/second box footprint.
    mask = terrain_reset_mask(roots, bodies, boxes, margin=0.4)
    assert mask.tolist() == [False, False, True]
    assert terrain_reset_mask(roots, bodies, ()).tolist() == [True, True, True]


def test_reset_candidates_remove_obstacle_frames_and_last_frame_of_each_clip():
    safe = torch.tensor([True, False, True, True, True, False, True])
    table, counts = build_reset_candidates(torch.tensor([0, 4]), torch.tensor([3, 6]), safe, bin_size=4)
    assert counts.tolist() == [2, 1]
    assert table[0, :2].tolist() == [0, 2]
    assert table[1, :1].tolist() == [4]


def test_reset_candidate_search_fails_clearly_instead_of_retrying_forever():
    with pytest.raises(ValueError, match="No safe reference reset frames"):
        build_reset_candidates(torch.tensor([0]), torch.tensor([2]), torch.tensor([False, False, True]), bin_size=4)
