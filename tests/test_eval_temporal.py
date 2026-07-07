from eval.benchmarking.eval_temporal_metrics import (
    aggregate_method_concept,
    per_frame_metrics,
)


def test_per_frame_basic():
    m = per_frame_metrics([0.1, 0.2, 0.9, 0.3, 0.1])
    assert m["max_frame_acc"] == 0.9
    assert m["video_acc_max"] == 1
    assert m["video_acc_mean"] == 0
    # mean = 0.32, gap = 0.9 - 0.32 = 0.58
    assert abs(m["reactivation_gap"] - 0.58) < 1e-6


def test_per_frame_all_below_tau():
    m = per_frame_metrics([0.1, 0.2, 0.3, 0.4])
    assert m["video_acc_max"] == 0
    assert m["video_acc_mean"] == 0
    assert m["max_frame_acc"] == 0.4


def test_per_frame_all_above_tau():
    m = per_frame_metrics([0.6, 0.7, 0.8])
    assert m["video_acc_max"] == 1
    assert m["video_acc_mean"] == 1


def test_aggregate_reactivation_rate():
    """Both videos exhibit reactivation (mean=0, max=1) → rate = 1.0."""
    vids = [
        {"video_acc_mean": 0, "video_acc_max": 1, "per_frame": [0.1, 0.9, 0.1]},
        {"video_acc_mean": 0, "video_acc_max": 1, "per_frame": [0.2, 0.8, 0.2]},
    ]
    a = aggregate_method_concept(vids)
    assert a["reactivation_rate"] == 1.0
    assert a["unlearn_acc_max"] == 1.0
    assert a["unlearn_acc_mean"] == 0.0
    assert a["mean_max_gap"] == 1.0


def test_aggregate_no_reactivation():
    """All zero or all hit; reactivation_rate should be 0."""
    vids = [
        {"video_acc_mean": 1, "video_acc_max": 1, "per_frame": [0.9, 0.9, 0.9]},
        {"video_acc_mean": 0, "video_acc_max": 0, "per_frame": [0.1, 0.1, 0.1]},
    ]
    a = aggregate_method_concept(vids)
    assert a["reactivation_rate"] == 0.0


def test_aggregate_empty():
    a = aggregate_method_concept([])
    assert a["n_videos"] == 0
