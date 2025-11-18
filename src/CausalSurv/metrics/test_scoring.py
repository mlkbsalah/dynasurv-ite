import torch
import numpy as np

class ConcordanceIndex:
    """Concordance Index calculator for survival analysis."""
    
    def __call__(self, estimate: torch.Tensor, event: torch.Tensor, time: torch.Tensor):
        """
        Calculate Harrell's Concordance Index (C-index).
        
        Args:
            estimate: Risk scores, shape (n,). Higher values = higher risk.
            event: Event indicators, shape (n,). True/1 = event occurred.
            time: Event/censoring times, shape (n,).
        Returns:
            C-index value (float between 0 and 1).
            
        How it works:
            - For each pair of subjects (i, j):
            - If i had an event before j's time (comparable pair):
                - Concordant: risk[i] > risk[j]
                - Discordant: risk[i] < risk[j]
                - Tied: risk[i] = risk[j] (counts as 0.5)
            - C-index = (concordant + 0.5 * tied) / total_comparable_pairs
        """
        n = len(time)
        
        concordant = 0.0
        discordant = 0.0
        tied = 0.0
        
        for i in range(n):
            if not event[i]:
                continue
                
            for j in range(n):
                if time[j] <= time[i]:
                    continue
                

                if estimate[i] > estimate[j]:
                    concordant += 1
                elif estimate[i] < estimate[j]:
                    discordant += 1
                else:
                    tied += 1
        
        total = concordant + discordant + tied
        
        if total == 0:
            return 0.5
        
        return (concordant + 0.5 * tied) / total

def kaplan_meier_estimate(time: torch.Tensor, event: torch.Tensor) -> tuple:
    """Compute Kaplan-Meier estimate of survival function."""
    device = time.device
    n_samples = time.shape[0]
    
    sorted_indices = torch.argsort(time)
    time_sorted = time[sorted_indices]
    event_sorted = event[sorted_indices]
    
    unique_times, inverse_indices = torch.unique_consecutive(time_sorted, return_inverse=True)
    n_times = unique_times.shape[0]
    
    at_risk = torch.zeros(n_times, device=device)
    events = torch.zeros(n_times, device=device)
    
    for i in range(n_times):
        at_risk[i] = (time_sorted >= unique_times[i]).sum()
        events[i] = event_sorted[inverse_indices == i].sum()
    
    survival_prob = torch.ones(n_times, device=device)
    for i in range(n_times):
        if at_risk[i] > 0:
            survival_prob[i] = survival_prob[i-1] * (1 - events[i] / at_risk[i]) if i > 0 else (1 - events[i] / at_risk[i])
        else:
            survival_prob[i] = survival_prob[i-1] if i > 0 else 1.0
    
    return unique_times, survival_prob

def concordance_index(sa_pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor, 
                     time_bins: torch.Tensor, mask: torch.Tensor):
    """Compute IPCW-corrected concordance index for each line in the batch."""
    device = sa_pred.device
    batch_size, n_lines = sa_pred.shape[:2]
    c_index_values = torch.zeros(n_lines, device=device)
    
    bin_indices = torch.searchsorted(time_bins, time) - 1
    bin_indices = torch.clamp(bin_indices, 0, sa_pred.shape[2] - 1)
    
    survival_at_event_times = torch.zeros((batch_size, n_lines), device=device)
    for i in range(batch_size):
        for l in range(n_lines):
            if mask[i, l]:
                bin_idx = bin_indices[i, l].item()
                surv_prob = torch.prod(sa_pred[i, l, :bin_idx+1])
                survival_at_event_times[i, l] = surv_prob
    
    

            
    for l in range(n_lines):
        masked_indices = mask[:, l] == 1
        if masked_indices.sum() > 0:
            masked_time = time[masked_indices, l]
            masked_event = event[masked_indices, l].bool()
            masked_survival = survival_at_event_times[masked_indices, l]
            
            c_index = ConcordanceIndex()
            c_index_value = c_index(
                estimate=-torch.log(masked_survival + 1e-8),
                event=masked_event,
                time=masked_time,
            )
            c_index_values[l] = c_index_value
        print()
    return c_index_values, survival_at_event_times, bin_indices


def test_kaplan_meier_simple():
    """Test 1: Simple Kaplan-Meier with known outcome"""
    print("=" * 60)
    print("TEST 1: Simple Kaplan-Meier Estimate")
    print("=" * 60)
    
    time = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    event = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    
    times, survival = kaplan_meier_estimate(time, event)
    
    print(f"Input times: {time.tolist()}")
    print(f"Input events: {event.tolist()}")
    print(f"\nUnique event times: {times.tolist()}")
    print(f"Survival probabilities: {[f'{s:.4f}' for s in survival.tolist()]}")
    
    expected_survival = [0.8, 0.6, 0.6, 0.3, 0.3]
    print(f"\nExpected survival: {expected_survival}")
    print(f"Match: {np.allclose(survival.numpy(), expected_survival, atol=1e-6)}")
    print()


def test_kaplan_meier_all_events():
    """Test 2: All events (no censoring)"""
    print("=" * 60)
    print("TEST 2: All Events (No Censoring)")
    print("=" * 60)
    
    time = torch.tensor([1.0, 2.0, 3.0])
    event = torch.tensor([1.0, 1.0, 1.0])
    
    times, survival = kaplan_meier_estimate(time, event)
    
    print(f"Input times: {time.tolist()}")
    print(f"Input events: {event.tolist()}")
    print(f"\nUnique event times: {times.tolist()}")
    print(f"Survival probabilities: {[f'{s:.4f}' for s in survival.tolist()]}")
    
    expected = [2/3, 1/3, 0.0]
    print(f"\nExpected survival: {[f'{s:.4f}' for s in expected]}")
    print(f"Match: {np.allclose(survival.numpy(), expected, atol=1e-6)}")
    print()


def test_kaplan_meier_all_censored():
    """Test 3: All censored (no events)"""
    print("=" * 60)
    print("TEST 3: All Censored (No Events)")
    print("=" * 60)
    
    time = torch.tensor([1.0, 2.0, 3.0])
    event = torch.tensor([0.0, 0.0, 0.0])
    
    times, survival = kaplan_meier_estimate(time, event)
    
    print(f"Input times: {time.tolist()}")
    print(f"Input events: {event.tolist()}")
    print(f"\nUnique event times: {times.tolist()}")
    print(f"Survival probabilities: {[f'{s:.4f}' for s in survival.tolist()]}")
    
    expected = [1.0, 1.0, 1.0]
    print(f"\nExpected survival: {expected}")
    print(f"Match: {np.allclose(survival.numpy(), expected, atol=1e-6)}")
    print()


def test_kaplan_meier_tied_times():
    """Test 4: Tied event times"""
    print("=" * 60)
    print("TEST 4: Tied Event Times")
    print("=" * 60)
    
    time = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0])
    event = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0])
    
    times, survival = kaplan_meier_estimate(time, event)
    
    print(f"Input times: {time.tolist()}")
    print(f"Input events: {event.tolist()}")
    print(f"\nUnique event times: {times.tolist()}")
    print(f"Survival probabilities: {[f'{s:.4f}' for s in survival.tolist()]}")
    
    expected = [0.6, 0.4, 0.0]
    print(f"\nExpected survival: {expected}")
    print(f"Match: {np.allclose(survival.numpy(), expected, atol=1e-6)}")
    print()


def test_concordance_index_simple():
    """Test 5: Simple concordance index test"""
    print("=" * 60)
    print("TEST 5: Concordance Index (Perfect Predictions)")
    print("=" * 60)
    
    batch_size = 5
    n_lines = 1
    n_intervals = 10 
    
    sa_pred = torch.ones(batch_size, n_lines, n_intervals) * 0.95
    for i in range(batch_size):
        decay_factor = 0.9**(i+1) # Higher index = lower survival
        sa_pred[i, 0, :] = decay_factor

    
    time = torch.tensor([[5.0], [4.0], [3.0], [2.0], [1.0]])
    event = torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0]])
    
    interval_cut = torch.linspace(0, 6, n_intervals+1)
    mask = torch.ones(batch_size, n_lines, dtype=torch.bool) # All valid
    
    c_indices, survival_at_event_times, bin_indices = concordance_index(sa_pred, time, event, interval_cut, mask)
    
    print(f"\nTime bins cut: {[f'{t:.2f}' for t in interval_cut.tolist()]}")
    print(f"\nEvent times: {time.squeeze().tolist()}")
    print(f"Event indicators: {event.squeeze().tolist()}")
    print()
    for i in range(batch_size):
        bin_low = interval_cut[bin_indices[i,0]].item()
        bin_high = interval_cut[bin_indices[i,0]+1].item()
        # print(f"Predicted survival probabilities for each sample:\n{sa_pred[i, 0, :].cumprod(dim=-1)}")
        print(f" Sample {i+1}: event time = {time[i,0].item()}, in bin idx = {bin_indices[i,0].item()}, range = ({bin_low:.2f}, {bin_high:.2f}), survival = {survival_at_event_times[i,0].item():.4f}")
        print()

    for i in range(batch_size):
        print(f" Sample {i+1}: risk score = {-torch.log(survival_at_event_times[i,0] + 1e-8).item():.4f}, survival = {survival_at_event_times[i,0].item():.4f}") 
    
    risk_ranking_list = [-torch.log(survival_at_event_times[i,0] + 1e-8).item() for i in range(batch_size)]
    sorted_indices = np.argsort(risk_ranking_list)[::-1]
    print("\nRisk ranking (from highest to lowest risk):")
    for rank, idx in enumerate(sorted_indices):
        print(f" Rank {rank+1}: Sample {idx+1} with risk score = {risk_ranking_list[idx]:.4f}")


    print(f"\nC-index for line 0: {c_indices[0].item():.4f}")
    print(f"Expected: High value (>0.8) for good risk ordering")
    print()


def test_concordance_index_detailed():
    """Test 6: Detailed concordance index with multiple lines"""
    print("=" * 60)
    print("TEST 6: Concordance Index (Multiple Lines)")
    print("=" * 60)
    
    batch_size = 3
    n_lines = 2
    n_intervals = 10
    
    # Line 0: Good predictions (survival decreases with time)
    # Line 1: Poor predictions (random survival)
    sa_pred = torch.ones(batch_size, n_lines, n_intervals)
    
    # Correct conditional survival (line 0)
    for i in range(batch_size):
        decay_factor = 0.9**(i+1) # Higher index = lower survival
        sa_pred[i, 0, :] = decay_factor
    
    # Random conditional survival (line 1)
    torch.manual_seed(42)
    sa_pred[:, 1, :] = torch.rand(batch_size, n_intervals) * 0.5 + 0.5  # Random values between 0.5 and 1.0
    
    time = torch.tensor([[5.0, 2.0],
                        [4.0, 3.0],
                        [2.5, 4.5],
                        [1.0, 5.5]])
    
    time = time[:batch_size, :]

    event = torch.tensor([[1.0, 1.0],
                         [1.0, 1.0],
                         [1.0, 0.0],
                         [0.0, 1.0]])
    event = event[:batch_size, :]
    
    time_bins = torch.linspace(0, 6, n_intervals+1)
    mask = torch.ones(batch_size, n_lines, dtype=torch.bool)
    

    # for l in range(n_lines):
    #     for i in range(batch_size):
    #         cum_surv = torch.cumprod(sa_pred[i, l, :], dim=-1)
    #         print(f"Patient {i}, Line {l}, Predicted cumulative survival: {[f'{s:.4f}' for s in cum_surv.tolist()]}")
    #     print()

    c_indices, survival_at_event, bin_indices = concordance_index(sa_pred, time, event, time_bins, mask)

    print(f"\nTime bins cut: {[f'{t:.2f}' for t in time_bins.tolist()]}")    

    for l in range(n_lines):
        print(f"Line {l} ")
        for i in range(batch_size):
            bin_low = time_bins[bin_indices[i,l]].item()
            bin_high = time_bins[bin_indices[i,l]+1].item()
            cumsurv = torch.cumprod(sa_pred[i, l, :], dim=-1)
            print(f"   Predicted cumulative survival for Patient {i}: {[f'{s:.4f}' for s in cumsurv.tolist()]}")
            print(f"   Patient {i}: event time = {time[i,l].item()}, in bin idx = {bin_indices[i,l].item()}, range = ({bin_low:.2f}, {bin_high:.2f}), survival = {survival_at_event[i,l].item():.4f}, risk = {-torch.log(survival_at_event[i,l] + 1e-8).item():.4f}")
            print()
        print()


    print(f"\nLine 0 - Event times: {time[:, 0].tolist()}")
    print(f"Line 0 - Events: {event[:, 0].tolist()}")
    print(f"Line 0 - C-index: {c_indices[0].item():.4f}")
    
    print(f"\nLine 1 - Event times: {time[:, 1].tolist()}")
    print(f"Line 1 - Events: {event[:, 1].tolist()}")
    print(f"Line 1 - C-index: {c_indices[1].item():.4f}")
    
    print(f"\nExpected: Line 0 should have higher C-index than Line 1")
    print(f"Actual difference: {(c_indices[0] - c_indices[1]).item():.4f}")
    print()


def run_all_tests():
    """Run all test cases"""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS FOR SURVIVAL ANALYSIS FUNCTIONS")
    print("=" * 60 + "\n")
    
    # test_kaplan_meier_simple()
    # test_kaplan_meier_all_events()
    # test_kaplan_meier_all_censored()
    # test_kaplan_meier_tied_times()
    # test_concordance_index_simple()
    test_concordance_index_detailed()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()