import random
import torch
import torch.nn as nn

from collections import deque

import config


# ============================================================
# REPLAY MEMORY
#
# Only used by DQN.
# Parent evolution does not need replay memory for learning.
# ============================================================

memory = deque(
    maxlen=10000
)


# ============================================================
# GET EXPERIENCE
# ============================================================

def get_experience(index):

    if (
        index < 0
        or index >= len(memory)
    ):

        raise IndexError(
            f"Index {index} is out of bounds "
            f"for memory of size {len(memory)}"
        )

    return memory[index]


# ============================================================
# CLEAR MEMORY
# ============================================================

def clear_memory():

    memory.clear()


# ============================================================
# CHOOSE ACTION
# ============================================================

def choose_action(
    model,
    state,
    epsilon=0.0
):

    # --------------------------------------------------------
    # RANDOM EXPLORATION
    #
    # Used for DQN.
    #
    # Evolution mode passes epsilon=0,
    # so evolved models always use their own brain.
    # --------------------------------------------------------

    if random.random() < epsilon:

        return random.randint(
            0,
            2
        )


    # --------------------------------------------------------
    # CONVERT STATE TO TENSOR
    # --------------------------------------------------------

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # RUN NEURAL NETWORK
    # --------------------------------------------------------

    with torch.no_grad():

        q_values = model(
            state_tensor
        )


    # --------------------------------------------------------
    # CHOOSE HIGHEST OUTPUT
    # --------------------------------------------------------

    return torch.argmax(
        q_values
    ).item()


# ============================================================
# ACTION -> CAR CONTROLS
#
# CarBrain currently has 3 outputs.
# Therefore we keep exactly 3 actions.
# ============================================================

def action_to_controls(action):

    # --------------------------------------------------------
    # ACTION 0
    # FORWARD / STRAIGHT
    # --------------------------------------------------------

    if action == 0:

        return 1, 0


    # --------------------------------------------------------
    # ACTION 1
    # FORWARD / LEFT
    # --------------------------------------------------------

    elif action == 1:

        return 1, 1


    # --------------------------------------------------------
    # ACTION 2
    # FORWARD / RIGHT
    # --------------------------------------------------------

    elif action == 2:

        return 1, -1


    else:

        raise ValueError(
            f"Invalid action: {action}"
        )


# ============================================================
# STORE EXPERIENCE
#
# Used by DQN only.
# ============================================================

def remember(
    state,
    action,
    reward,
    next_state,
    done
):

    memory.append(
        (
            state,
            action,
            reward,
            next_state,
            done
        )
    )


# ============================================================
# TRAIN DQN MODEL
#
# THIS FUNCTION IS NOT USED BY PARENT EVOLUTION.
# ============================================================

def train_model(
    model,
    target_model,
    optimizer,
    batch_size=64,
    gamma=0.99
):

    # --------------------------------------------------------
    # WAIT UNTIL MEMORY HAS ENOUGH EXPERIENCES
    # --------------------------------------------------------

    if len(memory) < batch_size:

        return None


    # --------------------------------------------------------
    # RANDOM SAMPLE FROM REPLAY MEMORY
    # --------------------------------------------------------

    batch = random.sample(
        memory,
        batch_size
    )


    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []


    # --------------------------------------------------------
    # SEPARATE EXPERIENCES
    # --------------------------------------------------------

    for (
        state,
        action,
        reward,
        next_state,
        done
    ) in batch:

        states.append(
            state
        )

        actions.append(
            action
        )

        rewards.append(
            reward
        )

        next_states.append(
            next_state
        )

        dones.append(
            done
        )


    # --------------------------------------------------------
    # CONVERT TO TENSORS
    # --------------------------------------------------------

    states = torch.tensor(
        states,
        dtype=torch.float32
    )

    actions = torch.tensor(
        actions,
        dtype=torch.long
    )

    rewards = torch.tensor(
        rewards,
        dtype=torch.float32
    )

    next_states = torch.tensor(
        next_states,
        dtype=torch.float32
    )

    dones = torch.tensor(
        dones,
        dtype=torch.float32
    )


    # ========================================================
    # CURRENT Q VALUES
    # ========================================================

    all_q_values = model(
        states
    )


    current_q_values = (
        all_q_values.gather(
            1,
            actions.unsqueeze(1)
        )
        .squeeze(1)
    )


    # ========================================================
    # TARGET Q VALUES
    # ========================================================

    with torch.no_grad():

        next_q_values = target_model(
            next_states
        )


        best_next_q = next_q_values.max(
            dim=1
        ).values


        target_q_values = (
            rewards
            +
            gamma
            * best_next_q
            * (1 - dones)
        )


    # ========================================================
    # LOSS
    # ========================================================

    loss_function = nn.MSELoss()


    loss = loss_function(
        current_q_values,
        target_q_values
    )


    # ========================================================
    # UPDATE MODEL
    # ========================================================

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    return loss.item()