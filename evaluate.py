def offlineEvaluate(mab, arms, rewards, contexts, num_of_events=10000, nrounds=None):
    """
    Offline evaluation of a multi-armed bandit

    Arguments
    =========
    mab : instance of MAB

    arms : 1D int array, shape (nevents,)
        integer arm id for each event

    rewards : 1D float array, shape (nevents,)
        reward received for each event

    contexts : 2D float array, shape (nevents, mab.narms*nfeatures)
        contexts presented to the arms (stacked horizontally)
        for each event.

    nrounds : int, optional
        number of matching events to evaluate `mab` on.

    Returns
    =======
    out : 1D float array
        rewards for the matching events
    """
    assert (arms.shape == (num_of_events,)), "1d array and in range[1...map.narms]"
    assert (rewards.shape == (num_of_events,)), "must be 1d array"
    assert (nrounds > 0 or nrounds is None), "must be positive integer with default None"
    h0 = []  # History list
    R0 = []  # Total Payoff

    count = 0
    for event in range(num_of_events):
        # If reach required number of rounds then stop
        # If number of rounds not specified, then read untill last of events.
        if len(h0) == nrounds:
            break
        # Play an arm, but the tround is the number of history observed
        action = mab.play(len(h0) + 1, contexts[event])

        # If the chosen arm is equal to the arm in the log,
        # then record history and payoff, and also update the arm.
        if action == arms[event]:
            count += 1
            h0.append(event)
            R0.append(rewards[event])
            mab.update(arms[event], rewards[event], contexts[event])

    return R0
