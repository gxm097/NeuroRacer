import random as random
import math
import torch

from Brain import CarBrain
import config


def crossover(
    parent1,
    parent2,
    parent3,
    mutation_rate=0.05,
    mutation_strength=0.05
):

    mutation_rate = config.MUTATION_RATE
    mutation_strength = config.MUTATION_STRENGTH

    child = CarBrain()


    with torch.no_grad():

        for (
            child_param,
            parent1_param,
            parent2_param,
            parent3_param
        ) in zip(
            child.parameters(),
            parent1.parameters(),
            parent2.parameters(),
            parent3.parameters()
        ):

            # ================================================
            # CHOOSE ONE OF THREE PARENTS
            # ================================================

            parent_choice = torch.rand_like(
                child_param
            )


            # Start with Parent 3
            inherited = parent3_param.clone()


            # Parent 2
            inherited = torch.where(
                parent_choice < 0.66,
                parent2_param,
                inherited
            )


            # Parent 1
            inherited = torch.where(
                parent_choice < 0.33,
                parent1_param,
                inherited
            )


            # ================================================
            # DECIDE WHICH PARAMETERS MUTATE
            # ================================================

            mutation_mask = (
                torch.rand_like(child_param)
                < mutation_rate
            )


            # ================================================
            # CREATE SMALL RANDOM MUTATIONS
            # ================================================

            mutation = (
                torch.randn_like(child_param)
                * mutation_strength
            )


            # ================================================
            # APPLY MUTATION
            # ================================================

            final_values = torch.where(
                mutation_mask,
                inherited + mutation,
                inherited
            )


            # ================================================
            # COPY INTO CHILD
            # ================================================

            child_param.copy_(
                final_values
            )

    return child


def getBest_parents(memory):
    if len(memory) < 2:
        raise ValueError("Not enough experiences in memory to select parents.")

    sorted_experiences = sorted(
        memory,
        key=lambda exp: exp[2],  #reward is at index 2
        reverse=True
    )

    best_parent1 = sorted_experiences[0]
    best_parent2 = sorted_experiences[1]

    return best_parent1, best_parent2