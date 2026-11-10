"""Application-deadline policy; this does not change simulated time or actions."""


def next_apply_deadline(scheduled, applied, finished, lateness, *, recover_overrun=False):
    """Preserve the legacy clock, optionally avoiding extra idle after overrun.

    The recovery branch never schedules earlier than last application +20ms.
    Normal steps keep the existing anchored clock and lateness threshold.
    """
    deadline = applied + .02 if lateness > .002 else scheduled + .02
    if finished >= deadline:
        deadline = max(finished, applied + .02) if recover_overrun else finished + .02
    return deadline
