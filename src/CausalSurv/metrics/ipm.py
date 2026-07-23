from typing import Callable

import torch


def pairwise_ipm(
    latent_state: torch.Tensor,
    treatment_idx: torch.Tensor,
    mask: torch.Tensor,
    valid_treatments_per_line: dict[int, list[int]],
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    min_group_size: int,
    balance_group_sizes: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean integral probability metric over treatment pairs, per line.

    Measures how far apart the treatment groups sit in representation space. Penalising it
    is what makes the per-treatment heads comparable across arms (Shalit et al., 2017).

    Args:
        latent_state: (batch, n_lines, latent_dim)
        treatment_idx: (batch, n_lines) observed treatment per line
        mask: (batch, n_lines) 1 where the line is observed
        valid_treatments_per_line: arms with enough dataset-level support, per line
        metric: two-sample discrepancy, e.g. MMDLoss or EMDLoss
        min_group_size: a pair contributes only if both groups reach this size *within the
            batch*. Unrelated to the dataset-level arm filter in the datamodule.
        balance_group_sizes: subsample both groups to min(n_i, n_j). The MMD V-statistic is
            biased by O(1/n) and the bias differs between groups of unequal size, so an
            unbalanced pair reads as nonzero even for identical distributions.
        generator: RNG for the subsampling. Pass a dedicated one to keep this off the global
            stream, so that enabling the diagnostic does not perturb the training trajectory.

    Returns:
        (value, stats). `value` is a 0-dim tensor on latent_state.device, exactly 0.0 when no
        pair qualifies. `stats` carries pairs_used / pairs_skipped / lines_contributing, which
        are what distinguish a dead regulariser from a merely weak one.
    """
    total = torch.zeros((), dtype=latent_state.dtype, device=latent_state.device)
    pairs_used = 0
    pairs_skipped = 0
    lines_contributing = 0

    for line in range(latent_state.shape[1]):
        valid_mask = mask[:, line].bool()
        if not valid_mask.any():
            continue

        z_line = latent_state[valid_mask, line, :]
        t_line = treatment_idx[valid_mask, line]

        # Lines with no valid mask are absent from the dict, not present-and-empty.
        valid_treatments = valid_treatments_per_line.get(line, [])
        z_groups = {k: z_line[t_line == k] for k in valid_treatments}

        line_contributed = False
        for i, k_i in enumerate(valid_treatments):
            # i + 1: a group against itself is identically zero and would only dilute the mean.
            for k_j in valid_treatments[i + 1 :]:
                z_i, z_j = z_groups[k_i], z_groups[k_j]
                n_i, n_j = z_i.shape[0], z_j.shape[0]

                if n_i < min_group_size or n_j < min_group_size:
                    pairs_skipped += 1
                    continue

                if balance_group_sizes and n_i != n_j:
                    n = min(n_i, n_j)
                    # Drawn on CPU so a caller-supplied generator works on any device.
                    pick_i = torch.randperm(n_i, generator=generator)[:n].to(z_i.device)
                    pick_j = torch.randperm(n_j, generator=generator)[:n].to(z_j.device)
                    z_i, z_j = z_i[pick_i], z_j[pick_j]

                total = total + metric(z_i, z_j)
                pairs_used += 1
                line_contributed = True

        if line_contributed:
            lines_contributing += 1

    value = total / pairs_used if pairs_used > 0 else total
    stats = {
        "pairs_used": float(pairs_used),
        "pairs_skipped": float(pairs_skipped),
        "lines_contributing": float(lines_contributing),
    }
    return value, stats
