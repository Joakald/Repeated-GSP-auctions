"""
Author: Joakim Alderborn
Email: joakimalder@gmail.com
This version: 2026-03-06

This is the Python code used to solve the model in my paper
"Repeated generalised second-price auctions with inconsistent and asymmetric time preferences"

To understand the code, you can first read the full paper which is on my profile at SSRN.com.
Most functions have explanatory comments, which should be enough if you've read the paper.

The parameter values are set in the main function at the end of the file.

"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from time import perf_counter
import os
import numpy as np

# ============================================================
# Helpers
# ============================================================

def interp_1d(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    return np.interp(xq, x, y)


def ks_statistic_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Returns KS statistic for two 1d arrays x and y,
    which may be of different size
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return 0.0
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    z = np.sort(np.concatenate([x_sorted, y_sorted]))
    cdf_x = np.searchsorted(x_sorted, z, side="right") / x_sorted.size
    cdf_y = np.searchsorted(y_sorted, z, side="right") / y_sorted.size   
    return float(np.max(np.abs(cdf_x - cdf_y)))


def ks_distance_cutoffs_marginal_max(mu_a: np.ndarray, mu_b: np.ndarray) -> float:
    """
    Compare two cutoff-sample matrices mu_a, mu_b of shape (S,K).
    Returns max KS distance across the K marginals.
    """
    mu_a = np.asarray(mu_a, dtype=float)
    mu_b = np.asarray(mu_b, dtype=float)
    if mu_a.shape != mu_b.shape:
        raise ValueError("mu_a and mu_b must have the same shape (S,K).")
    S, K = mu_a.shape
    dmax = 0.0
    for k in range(K):
        dmax = max(dmax, ks_statistic_1d(mu_a[:, k], mu_b[:, k]))
    return float(dmax)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def compute_sorted_and_rank(bids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes:
        bids[i] = the bid of player i
    Returns:
        sorted_bids give the bids in descending order
        rank[i] = the rank of player i in the bid order (starts at rank 0)
    """
    bids = np.asarray(bids, dtype=float)
    order = np.argsort(-bids)  # descending
    sorted_bids = bids[order]
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return sorted_bids, rank

def bidder_cutoffs_from_sorted(sorted_bids: np.ndarray, rank: np.ndarray, i: int, K: int) -> np.ndarray:
    """
    For player i and for a set of sorted bids, calculate the
    cutoffs faced by the player. That is, remove player i from the ranking,
    and among the remaining players get the bids of the players who occupy the
    top K positions. Since we assume that N > K, there will always be enough
    players to fill all K slots.
    """
    N = rank.size
    r = int(rank[i])
    # Build "others" sorted list without i
    others = np.empty(N - 1, dtype=float)
    if r > 0:
        others[:r] = sorted_bids[:r]
    if r < N - 1:
        others[r:] = sorted_bids[r + 1 :]
    # Cutoffs are top K of others
    cut = others[:K].copy()
    return cut

def gsp_outcome_given_cutoffs(bid: float, cutoffs: np.ndarray) -> Tuple[Optional[int], float]:
    """
    Given a single bid and a set of K cutoffs, find the position a player would
    obtain with that bid. Return the position and the cutoff that you beat, which would
    be the price you pay. If no slot is obtained (i.e. the bid is smaller than all the cutoffs),
    return (None,0.0), meaning no position and zero price.
    """
    for k, c in enumerate(cutoffs):
        if bid >= c:
            return k, float(c)
    return None, 0.0


# ============================================================
# Model Config Class
# ============================================================

@dataclass
class ModelConfig:
    """
    This class holds everything that the user needs to set.
    We use no defaults, so the constructor must take every variable

    Estimation player are draws randomly from the set of admissible parameters
    Evalution player are defined explicitly by user
    """

    # Number of players, number of slots, slot click rates
    N: int
    K: int
    alpha: Tuple[float, ...] # Note: size of alpha must equal K

    # Integer grid bounds for budget and bid
    B_min: int
    B_max: int
    b_min: int
    b_max: int

    # Budgets dynamics
    budget_growth: float
    income: float

    # Admissible values, betas and deltas
    v_set: Tuple[float, ...]
    beta_set: Tuple[float, ...]
    delta_set: Tuple[float, ...]

    # Estimation tools
    eval_max_iter: int      # maximum iterations for auxiliary function
    tol_W: float            # tolerance for convergence of auxiliary function
    tol_policy_max: float   # tolerance for maximum policy deviation
    tol_policy_mean: float  # tolerance for mean policy deviation
    outer_max_iter: int     # maximum iterations for main (outer) loop in solve()
    tol_mu: float           # tolerance for sample (mu) KS statistics
    mu_damping: float       # fraction of cutoffs that are being updated each iteration
    S: int                  # number of rows (or sets of cutoffs) in matrix of cutoff (mu) samples
    sim_T: int              # number of iteration when estimating cutoffs samples from policy
    mc_C: int               # number of draws from cutoffs samples used to calculate expectation in value function
    burn_in: int            # number of iterations in final simulation before logging starts
    T_eval: int             # number of iterations of logging in final simulation

    # Evaluation players used in final simulation 
    # Must be given as explicit indexes for value, beta, delta and budget: (iv, ibeta, idelta, iB0)
    # For example: (0, 1, 2, 25) defines a player
    # If None, evaluation will reuse estimation players.
    eval_players_idx: Optional[List[Tuple[int, int, int, int]]]

    # Output persistence
    output_dir: str
    run_name: str
    save_npz: bool
    save_txt: bool

    # RNG
    rng_seed: int


# ============================================================
# Solver class
# ============================================================

class DynamicGSPSolver:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.rng_seed)

        # Checks
        if cfg.N <= cfg.K:
            raise ValueError("Assumption required: N > K.")
        if len(cfg.alpha) != cfg.K:
            raise ValueError("alpha must have length K.")
        if not (0 <= cfg.B_min <= cfg.B_max):
            raise ValueError("Require 0 <= B_min <= B_max.")
        if not (0 <= cfg.b_min <= cfg.b_max):
            raise ValueError("Require 0 <= b_min <= b_max.")
        if cfg.S <= 0 or cfg.sim_T <= 0 or cfg.mc_C <= 0:
            raise ValueError("Require S > 0, sim_T > 0, mc_C > 0.")
        if not (0.0 <= cfg.mu_damping <= 1.0):
            raise ValueError("mu_damping must be in [0,1].")

        if len(cfg.v_set) == 0:
            raise ValueError("v_set must be non-empty.")
        if len(cfg.beta_set) == 0 or len(cfg.delta_set) == 0:
            raise ValueError("beta_set and delta_set must be non-empty.")
        if any((b <= 0.0 or b > 1.0) for b in cfg.beta_set):
            raise ValueError("beta_set must be in (0,1].")
        if any((d <= 0.0 or d > 1.0) for d in cfg.delta_set):
            raise ValueError("delta_set must be in (0,1].")

        # Alpha restriction (for budget feasibility safety)
        if any((a > 1.0) for a in cfg.alpha):
            raise ValueError("All click rates alpha_k must satisfy alpha_k <= 1.")

        # Store arrays
        self.alpha = np.asarray(cfg.alpha, dtype=float)
        self.B_grid = np.arange(cfg.B_min, cfg.B_max + 1, dtype=float)
        self.b_grid = np.arange(cfg.b_min, cfg.b_max + 1, dtype=float)
        self.v_set = np.asarray(cfg.v_set, dtype=float)
        self.beta_set = np.asarray(cfg.beta_set, dtype=float)
        self.delta_set = np.asarray(cfg.delta_set, dtype=float)

        self.n_B = self.B_grid.size
        self.n_b = self.b_grid.size
        self.n_v = self.v_set.size
        self.n_beta = self.beta_set.size
        self.n_delta = self.delta_set.size

        # Main DP objects: W_aux and policy defined over indices (beta, delta, value, budget)
        self.W_aux = np.zeros((self.n_beta, self.n_delta, self.n_v, self.n_B), dtype=float)
        self.policy = np.zeros((self.n_beta, self.n_delta, self.n_v, self.n_B), dtype=float)

        # mu samples: (S,K)
        # this is our estimation of the joint probability distribution of the cutoffs
        self.mu_samples: Optional[np.ndarray] = None

        # Estimation player primitives (set in solve)
        self.est_v_idx: Optional[np.ndarray] = None
        self.est_B0: Optional[np.ndarray] = None
        self.est_beta_idx: Optional[np.ndarray] = None
        self.est_delta_idx: Optional[np.ndarray] = None

    # ============================================================
    # Player initialization
    # ============================================================

    def _init_estimation_players(self) -> None:
        """
        Draw N estimation players as indices from admissible sets and budget grid.
        """
        cfg = self.cfg
        self.est_v_idx = self.rng.integers(0, self.n_v, size=cfg.N, dtype=int)
        self.est_beta_idx = self.rng.integers(0, self.n_beta, size=cfg.N, dtype=int)
        self.est_delta_idx = self.rng.integers(0, self.n_delta, size=cfg.N, dtype=int)
        B0_idx = self.rng.integers(0, self.n_B, size=cfg.N, dtype=int)
        self.est_B0 = self.B_grid[B0_idx].astype(float)

    def _build_eval_players(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build evaluation players arrays:
          B0_eval (N_eval,), v_idx_eval, beta_idx_eval, delta_idx_eval.
        If cfg.eval_players_idx is None, reuse estimation players.
        """
        cfg = self.cfg
        if cfg.eval_players_idx is None:
            assert self.est_B0 is not None
            assert self.est_v_idx is not None
            assert self.est_beta_idx is not None
            assert self.est_delta_idx is not None
            return (self.est_B0.copy(), self.est_v_idx.copy(), self.est_beta_idx.copy(), self.est_delta_idx.copy())

        idx_list = list(cfg.eval_players_idx)
        if len(idx_list) == 0:
            raise ValueError("eval_players_idx is an empty list; provide at least one player tuple.")

        B0 = np.empty(len(idx_list), dtype=float)
        v_idx = np.empty(len(idx_list), dtype=int)
        beta_idx = np.empty(len(idx_list), dtype=int)
        delta_idx = np.empty(len(idx_list), dtype=int)

        for i, (iv, ib, idl, iB0) in enumerate(idx_list):
            if not (0 <= iv < self.n_v):
                raise ValueError(f"eval player {i} has iv={iv} out of range [0,{self.n_v-1}].")
            if not (0 <= ib < self.n_beta):
                raise ValueError(f"eval player {i} has ibeta={ib} out of range [0,{self.n_beta-1}].")
            if not (0 <= idl < self.n_delta):
                raise ValueError(f"eval player {i} has idelta={idl} out of range [0,{self.n_delta-1}].")
            if not (0 <= iB0 < self.n_B):
                raise ValueError(f"eval player {i} has iB0={iB0} out of range [0,{self.n_B-1}].")

            v_idx[i] = iv
            beta_idx[i] = ib
            delta_idx[i] = idl
            B0[i] = float(self.B_grid[iB0])

        return B0, v_idx, beta_idx, delta_idx

    # ============================================================
    # Budget transition
    # ============================================================

    def _budget_transition(self, B: np.ndarray, spend: np.ndarray) -> np.ndarray:
        """
        Standard budget dynamics: add income, remove spending, apply growth factor.
        Clip to ensure that budget stays within admissible set.
        """
        cfg = self.cfg
        Bp = (1.0 + cfg.budget_growth) * (B + cfg.income - spend)
        return np.clip(Bp, float(cfg.B_min), float(cfg.B_max))

    # ============================================================
    # mu initialization and sampling
    # ============================================================

    def _initialize_mu(self) -> None:
        """
        Set up initial cutoff samples.
        We randomly pick from the bid grip as an initial guess.
        """
        cfg = self.cfg
        mu = self.rng.integers(cfg.b_min, cfg.b_max + 1, size=(cfg.S, cfg.K)).astype(float)
        mu.sort(axis=1)
        mu = mu[:, ::-1] # invert every row
        self.mu_samples = mu

    def _draw_cutoffs_from_mu(self, n: int) -> np.ndarray:
        """
        Randomly draw n cutoff rows to get a (n,K) matrix by sampling rows with replacement
        from cutoff samples.
        """
        assert self.mu_samples is not None
        idx = self.rng.integers(0, self.mu_samples.shape[0], size=n)
        return self.mu_samples[idx].copy()

    # ============================================================
    # Precompute bid outcomes for give cutoff draws
    # ============================================================

    def _precompute_bid_outcomes(self, C_draws: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For every draw in cutoffs draws (mc_C,K) and every bid in the bid grid (jb,),
        compute for that draw and that bid what outcome a player with that bid would obtian.
        Returns three (jb,mc_C) with the click rate, price and spending obtained.
        """
        C_draws = np.asarray(C_draws, dtype=float)
        if C_draws.ndim != 2 or C_draws.shape[1] != self.cfg.K:
            raise ValueError("C_draws must have shape (mc_C, K).")
        mc_C = C_draws.shape[0]

        click = np.zeros((self.n_b, mc_C), dtype=float)
        price = np.zeros((self.n_b, mc_C), dtype=float)
        spend = np.zeros((self.n_b, mc_C), dtype=float)

        for jb, b in enumerate(self.b_grid):
            # For each cutoff draw, determine slot and price
            for c in range(mc_C):
                slot, p = gsp_outcome_given_cutoffs(float(b), C_draws[c])
                if slot is not None:
                    cr = float(self.alpha[slot])
                    click[jb, c] = cr
                    price[jb, c] = p
                    spend[jb, c] = cr * p

        return click, price, spend

    # ============================================================
    # Inner solve function
    # ============================================================

    def _solve_inner(self) -> Dict[str, Any]:
        """
        Given current estimation of cutoffs, find and set policy and auxiliary function. 
            1. For each (ibeta, idelta, iv, iB) choose bid maximizing E[u + beta*delta*W(B')]
            2. Set up a loop for DP equation W(B) = E[u + delta*W(B')] to make W converge
        Returns some data for logging.
        """
        cfg = self.cfg
        assert self.mu_samples is not None and self.mu_samples.shape == (cfg.S, cfg.K)
        t0 = perf_counter()

        # Draw cutoffs and get outcomes for every bid
        C_draws = self._draw_cutoffs_from_mu(cfg.mc_C)
        click, price, spend = self._precompute_bid_outcomes(C_draws)

        # -----------------
        # 1. GET POLICY
        # -----------------

        t_pol = perf_counter()
        W_old = self.W_aux.copy()
        policy_old = self.policy.copy()
        new_policy = np.empty_like(self.policy)
        W_greedy = np.empty_like(self.W_aux)
        # single iterator over the full 4D state space (ibeta, idelta, iv, iB)
        for ib, idl, iv, iB in np.ndindex(self.n_beta, self.n_delta, self.n_v, self.n_B):

            beta = float(self.beta_set[ib])
            delta = float(self.delta_set[idl])
            v = float(self.v_set[iv])
            B = float(self.B_grid[iB])

            # continuation slice is fixed on (ib, idl, iv)
            W_slice = W_old[ib, idl, iv]  # shape (n_B,)

            best_obj = -np.inf
            best_bid = float(self.b_grid[0])

            for jb, b in enumerate(self.b_grid):
                """
                Here are two types of restrictions:
                1. Break if, for the given bid jb, there exists a draw such that what
                I would have to pay exceeds my budget. This means sometimes, I am allowed to
                bid more than my budget.
                2. Never allow bids above budget.
                """
                if np.max(spend[jb,:]) > B:
                    continue
                
                #if b > B:  # can't bid more money than we have
                #   break
                
                u = click[jb] * (v - price[jb])
                Bp = self._budget_transition(B, spend[jb])
                Wp = interp_1d(self.B_grid, W_slice, Bp)
                # objective is the value function
                obj = float(np.mean(u + beta * delta * Wp))

                if obj > best_obj:
                    best_obj = obj
                    best_bid = float(b)

            new_policy[ib, idl, iv, iB] = best_bid
            W_greedy[ib, idl, iv, iB] = best_obj

        self.policy = new_policy
        max_diff_policy = float(np.max(np.abs(self.policy - policy_old)))
        min_diff_policy = float(np.min(np.abs(self.policy - policy_old)))
        mean_diff_policy = float(np.mean(np.abs(self.policy - policy_old)))
        max_policy = float(np.max(self.policy))
        min_policy = float(np.min(self.policy))
        mean_policy = float(np.mean(self.policy))

        """
        print(self.policy.shape)
        print(self.W_aux.shape)
        print(np.mean(self.policy))
        print(np.min(self.policy))
        print(np.max(self.policy))
        """

        print(f"   Policy finished in ", perf_counter() - t_pol,"seconds.")
        print(f"      Policy Diff Max =  {max_diff_policy:.3f}")
        print(f"      Policy Diff Min =  {min_diff_policy:.3f}")
        print(f"      Policy Diff Mean = {mean_diff_policy:.3f}")
        print(f"      Policy Max =  {max_policy:.3f}")
        print(f"      Policy Min =  {min_policy:.3f}")
        print(f"      Policy Mean = {mean_policy:.3f}")

        # --------------------------
        # 2. POLICY EVALUATION
        # --------------------------
        
        t_eval = perf_counter()       
        W = W_old.copy()
        eval_iters = 0
        diff_W = np.inf

        # make sure we are on grid
        def bid_to_jb(bid: float) -> int:
            jb = int(round(bid - cfg.b_min))
            if jb < 0:
                return 0
            if jb >= self.n_b:
                return self.n_b - 1
            return jb
        policy_jb = np.empty_like(self.policy, dtype=int)
        for ib, idl, iv, iB in np.ndindex(self.n_beta, self.n_delta, self.n_v, self.n_B):
            policy_jb[ib, idl, iv, iB] = bid_to_jb(float(self.policy[ib, idl, iv, iB]))

        # Evaluate DP expectation under fixed policy
        for it in range(cfg.eval_max_iter):
            eval_iters = it + 1
            W_new = np.empty_like(W)

            for ib, idl, iv, iB in np.ndindex(self.n_beta, self.n_delta, self.n_v, self.n_B):
                
                delta = float(self.delta_set[idl])
                v = float(self.v_set[iv])
                B = float(self.B_grid[iB])

                W_slice = W[ib, idl, iv,:]
                jb = int(policy_jb[ib, idl, iv, iB])

                u = click[jb] * (v - price[jb])
                Bp = self._budget_transition(B, spend[jb])
                Wp = interp_1d(self.B_grid, W_slice, Bp)
                W_new[ib, idl, iv, iB] = float(np.mean(u + delta * Wp))

            diff_W = float(np.max(np.abs(W_new - W)))
            W = W_new
            if diff_W <= cfg.tol_W:
                break

        self.W_aux = W

        max_W_aux = float(np.max(self.W_aux))
        min_W_aux = float(np.min(self.W_aux))
        mean_W_aux = float(np.mean(self.W_aux))

        print(f"   Evaluation finished in ", perf_counter()-t_eval,"seconds.")
        print(f"      W_aux Max = {max_W_aux:.3f}")
        print(f"      W_aux Min = {min_W_aux:.3f}")
        print(f"      W_aux Mean = {mean_W_aux:.3f}")

        t1 = perf_counter()
        return {
            "inner_time_sec": float(t1 - t0),
            "diff_policy": max_diff_policy,
            "mean_diff_policy": mean_diff_policy,
            "diff_W": diff_W,
            "eval_iters": int(eval_iters),
        }

    # ============================================================
    # Policy application
    # ============================================================

    def _bids_from_policy(self, B: np.ndarray, v_idx: np.ndarray, beta_idx: np.ndarray, delta_idx: np.ndarray) -> np.ndarray:
        """
        For given set up player profiles (budget, value, beta, delta),
        use the policy rule to compute bids for every player
        """
        B = np.asarray(B, dtype=float).reshape(-1)
        v_idx = np.asarray(v_idx, dtype=int).reshape(-1)
        beta_idx = np.asarray(beta_idx, dtype=int).reshape(-1)
        delta_idx = np.asarray(delta_idx, dtype=int).reshape(-1)
        N = B.size

        bids = np.empty(N, dtype=float)
        for i in range(N):
            ib = int(beta_idx[i])
            idl = int(delta_idx[i])
            iv = int(v_idx[i])
            # 1D policy slice over budgets
            pol_slice = self.policy[ib, idl, iv,:]  # (n_B,)
            b = float(interp_1d(self.B_grid, pol_slice, np.array([B[i]], dtype=float))[0])
            # clamp to feasibility
            if b > B[i]:
                b = float(B[i])
            if b < self.cfg.b_min:
                b = float(self.cfg.b_min)
            bids[i] = b
        return bids

    # ============================================================
    # Welfare for subgroups
    # ============================================================

    def _compute_welfare_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate average welfare stats for all types of players over the evalution,
        using the results obtained from the final simulation.
        """

        welfare_dyn = np.asarray(results["welfare_dyn"], dtype=float)
        winners_idx = results["winners_idx_dyn"]
        eval_v_idx = results["eval_v_idx"]
        eval_beta_idx = results["eval_beta_idx"]
        eval_delta_idx = results["eval_delta_idx"]

        T_eval = welfare_dyn.size
        half = T_eval // 2

        stats = {}

        # Aggregate statistics
        stats["welfare_mean_all"] = float(np.mean(welfare_dyn))
        stats["welfare_std"] = float(np.std(welfare_dyn))
        stats["welfare_mean_first_half"] = float(np.mean(welfare_dyn[:half]))
        stats["welfare_mean_second_half"] = float(np.mean(welfare_dyn[half:]))

        # Type-level contributions
        type_contrib = {}

        for iv in range(self.n_v):
            for ib in range(self.n_beta):
                for idl in range(self.n_delta):
                    key = (iv, ib, idl)
                    type_contrib[key] = 0.0

        for t in range(T_eval):
            for k in range(self.cfg.K):
                pid = winners_idx[t, k]
                iv = eval_v_idx[pid]
                ib = eval_beta_idx[pid]
                idl = eval_delta_idx[pid]
                v = self.v_set[iv]
                type_contrib[(iv, ib, idl)] += self.alpha[k] * v

        # Convert to per-period averages
        for key in type_contrib:
            type_contrib[key] /= T_eval

        stats["welfare_by_type"] = type_contrib
        return stats

    # ============================================================
    # Outer loop: simulate cutoffs under policy and update mu
    # ============================================================

    def _simulate_pool_cutoffs_and_budgets(
        self,
        B0: np.ndarray,
        v_idx: np.ndarray,
        beta_idx: np.ndarray,
        delta_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This is the simulation at the end of each iteration in the estimation. 
        Use the policy rules obtained in the current iteration to simulate sim_T periods 
        with the estimation players and store the resulting cutoff estimates.
        Returns:
          raw_cutoffs: (sim_T*S, K)
          B_end: (N_players,)
        """
        cfg = self.cfg
        B = np.asarray(B0, dtype=float).copy()
        v_idx = np.asarray(v_idx, dtype=int).copy()
        beta_idx = np.asarray(beta_idx, dtype=int).copy()
        delta_idx = np.asarray(delta_idx, dtype=int).copy()

        N = B.size
        K, S, T = cfg.K, cfg.S, cfg.sim_T

        raw = np.empty((T * S, K), dtype=float)
        write = 0

        for _t in range(T):
            bids = self._bids_from_policy(B, v_idx, beta_idx, delta_idx)
            sorted_bids, rank = compute_sorted_and_rank(bids)

            idxs = self.rng.integers(0, N, size=S)
            for i in idxs:
                raw[write, :] = bidder_cutoffs_from_sorted(sorted_bids, rank, int(i), K)
                write += 1

            winners = np.argsort(rank)[:K]

            # Prices: next-highest bid, last pays 0
            prices = np.empty(K, dtype=float)
            for k in range(K):
                prices[k] = sorted_bids[k + 1] if k < K - 1 else 0.0

            spend = np.zeros(N, dtype=float)
            for k, i in enumerate(winners):
                spend[i] = float(self.alpha[k]) * float(prices[k])

            B = self._budget_transition(B, spend)

        return raw, B

    def _resample_mu_from_pool(self, raw_cutoffs: np.ndarray) -> np.ndarray:
        """
        Having obtained the cutoffs (sim_T*S,K) from the simulation,
        draw S cutoffs to use in the next round.
        """
        cfg = self.cfg
        raw = np.asarray(raw_cutoffs, dtype=float)
        idx = self.rng.integers(0, raw.shape[0], size=cfg.S)
        mu_new = raw[idx].copy()
        mu_new.sort(axis=1)
        mu_new = mu_new[:, ::-1]
        return mu_new

    # ============================================================
    # Benchmarks and evaluation logging
    # ============================================================

    def _vcg_allocation(self, v_idx: np.ndarray) -> np.ndarray:
        """
        Calculate highest possible welfare by assigning slots in order or value.
        This is what would be obtained by the static VCG mechanism.
        """
        v = self.v_set[np.asarray(v_idx, dtype=int)]
        return np.argsort(-v)[: self.cfg.K].astype(int)

    def _simulate_and_log_allocations(
        self,
        B0: np.ndarray,
        v_idx: np.ndarray,
        beta_idx: np.ndarray,
        delta_idx: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        This is the final simulation. Use the policy obtained in the estimation to run
        repeated GSP auctions with budget update in between to make them connected, 
        and store results from each auction.
        Returns a bunch of statistics that we can use to check how the model behaves...
        """
        
        cfg = self.cfg
        B = np.asarray(B0, dtype=float).copy()
        v_idx = np.asarray(v_idx, dtype=int).copy()
        beta_idx = np.asarray(beta_idx, dtype=int).copy()
        delta_idx = np.asarray(delta_idx, dtype=int).copy()

        N = B.size
        K = cfg.K

        winners_idx_dyn = np.empty((cfg.T_eval, K), dtype=int)
        welfare_dyn = np.empty(cfg.T_eval, dtype=float)
        bids_winners_dyn = np.empty((cfg.T_eval, K), dtype=float)
        budgets_winners_dyn = np.empty((cfg.T_eval, K), dtype=float)
        prices_winners_dyn = np.empty((cfg.T_eval, K), dtype=float)
        budgets_eval_path = np.empty((cfg.T_eval, N), dtype=float)
        bids_eval_path = np.empty((cfg.T_eval, N), dtype=float)

        eval_ptr = 0
        total_T = cfg.burn_in + cfg.T_eval

        for t in range(total_T):

            bids = self._bids_from_policy(B, v_idx, beta_idx, delta_idx)

            noise = 1e-12 * self.rng.standard_normal(size=N)
            order = np.argsort(-(bids + noise))
            winners = order[:K]
            sorted_bids_desc = bids[order]

            prices = np.zeros(K, dtype=float)
            for k in range(K):
                prices[k] = float(sorted_bids_desc[k + 1])

            spend = np.zeros(N, dtype=float)
            for k, i in enumerate(winners):
                spend[i] = float(self.alpha[k]) * float(prices[k])

            # Logging evaluation stats (pre-spend budgets B)
            if t >= cfg.burn_in:
                winners_idx_dyn[eval_ptr, :] = winners
                bids_winners_dyn[eval_ptr, :] = bids[winners]
                budgets_winners_dyn[eval_ptr, :] = B[winners]
                prices_winners_dyn[eval_ptr, :] = prices
                budgets_eval_path[eval_ptr, :] = B.copy()
                bids_eval_path[eval_ptr, :] = bids.copy()

                v_vals = self.v_set[v_idx[winners]]
                welfare_dyn[eval_ptr] = float(np.sum(self.alpha * v_vals))

                eval_ptr += 1
                if eval_ptr >= cfg.T_eval:
                    B = self._budget_transition(B, spend)
                    break

            # State transition
            B = self._budget_transition(B, spend)

        return {
            "winners_idx_dyn": winners_idx_dyn,
            "welfare_dyn": welfare_dyn,
            "bids_winners_dyn": bids_winners_dyn,
            "prices_winners_dyn": prices_winners_dyn,
            "budgets_winners_dyn": budgets_winners_dyn,
            "B_eval_start": np.asarray(B0, dtype=float).copy(),
            "budgets_path_dyn": budgets_eval_path,
            "bids_path_dyn": bids_eval_path,
        }

    # ============================================================
    # Output
    # ============================================================

    def _run_output_path(self) -> str:
        cfg = self.cfg
        run_name = cfg.run_name.strip()
        if run_name == "":
            run_name = f"run_seed_{cfg.rng_seed}"
        path = os.path.join(cfg.output_dir, run_name)
        _ensure_dir(path)
        return path

    def _fmt_row(self, cols, widths, aligns):
        """
        Row formatter.
        """
        out = []
        for val, w, a in zip(cols, widths, aligns):
            s = str(val)

            if w <= 0:
                out.append(s)
                continue

            if len(s) > w:
                s = s[:w]
            out.append(f"{s:{a}{w}}")

        return " ".join(out)

    def _write_budget_bid_history_txt(self, results: Dict[str, Any], out_path: str) -> None:

        budgets = np.asarray(results["budgets_path_dyn"], dtype=float)  # (T_eval, N_eval)
        bids    = np.asarray(results["bids_path_dyn"], dtype=float)     # (T_eval, N_eval)
        welfare = np.asarray(results["welfare_dyn"], dtype=float)       # (T_eval,)

        prices_winners = np.asarray(results["prices_winners_dyn"], dtype=float)  # (T_eval, K)

        T_eval, N_eval = budgets.shape
        if bids.shape != (T_eval, N_eval):
            raise ValueError(f"bids_path_dyn shape {bids.shape} does not match budgets_path_dyn {budgets.shape}")
        if welfare.shape != (T_eval,):
            raise ValueError(f"welfare_dyn shape {welfare.shape} does not match T_eval={T_eval}")
        if prices_winners.shape != (T_eval, self.cfg.K):
            raise ValueError(f"prices_winners_dyn shape {prices_winners.shape} != ({T_eval},{self.cfg.K})")

        revenue = np.sum(prices_winners * self.alpha.reshape(1, -1), axis=1)  # (T_eval,)

        with open(out_path, "w", encoding="utf-8") as f:
            cols = (
                ["t", "revenue", "welfare"]
                + [f"B_p{p}" for p in range(N_eval)]
                + [f"bid_p{p}" for p in range(N_eval)]
            )
            f.write(" ".join(cols) + "\n")

            for t in range(T_eval):
                row = [str(t), f"{revenue[t]:.6f}", f"{welfare[t]:.6f}"]
                row += [f"{budgets[t, p]:.6f}" for p in range(N_eval)]
                row += [f"{bids[t, p]:.6f}"    for p in range(N_eval)]
                f.write(" ".join(row) + "\n")

    def _write_policy_by_type_txt(self, out_path: str) -> None:

        B = self.B_grid

        # Header
        col_names = ["B"]
        for iv in range(self.n_v):
            for ib in range(self.n_beta):
                for idl in range(self.n_delta):
                    col_names.append(f"bid_iv{iv}_ib{ib}_idl{idl}")
        
        # data
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(" ".join(col_names) + "\n")

            for iB in range(self.n_B):
                row = [f"{float(B[iB]):.6f}"]
                for iv in range(self.n_v):
                    for ib in range(self.n_beta):
                        for idl in range(self.n_delta):
                            bid = float(self.policy[ib, idl, iv, iB])
                            row.append(f"{bid:.6f}")
                f.write(" ".join(row) + "\n")

    def _write_run_summary_txt(self, results: Dict[str, Any], out_path: str) -> None:
        cfg: ModelConfig = results["config"]

        # Estimation players
        est_v_idx = np.asarray(results["est_v_idx"], dtype=int)
        est_B0 = np.asarray(results["est_B0"], dtype=float)
        est_beta_idx = np.asarray(results["est_beta_idx"], dtype=int)
        est_delta_idx = np.asarray(results["est_delta_idx"], dtype=int)

        # Evaluation players
        eval_v_idx = np.asarray(results["eval_v_idx"], dtype=int)
        eval_B0 = np.asarray(results["eval_B0"], dtype=float)
        eval_beta_idx = np.asarray(results["eval_beta_idx"], dtype=int)
        eval_delta_idx = np.asarray(results["eval_delta_idx"], dtype=int)
        N_eval = int(eval_B0.size)

        # calculate welfare
        welfare_vcg = float(results["welfare_vcg"])
        welfare_dyn = np.asarray(results["welfare_dyn"], dtype=float)
        T_eval = int(welfare_dyn.size)
        half = T_eval // 2

        welfare_gsp_mean = float(np.mean(welfare_dyn)) if T_eval > 0 else np.nan
        welfare_gsp_std = float(np.std(welfare_dyn)) if T_eval > 0 else np.nan
        welfare_gap = float(welfare_vcg - welfare_gsp_mean) if T_eval > 0 else np.nan
        welfare_ratio = float(welfare_gsp_mean / welfare_vcg) if welfare_vcg != 0.0 and T_eval > 0 else np.nan

        welfare_first_half = float(np.mean(welfare_dyn[:half])) if half > 0 else welfare_gsp_mean
        welfare_second_half = float(np.mean(welfare_dyn[half:])) if half > 0 else welfare_gsp_mean
        welfare_min = float(np.min(welfare_dyn)) if T_eval > 0 else np.nan
        welfare_max = float(np.max(welfare_dyn)) if T_eval > 0 else np.nan

        winners_idx_dyn = np.asarray(results["winners_idx_dyn"], dtype=int)
        budgets_winners_dyn = np.asarray(results["budgets_winners_dyn"], dtype=float)
        bids_winners_dyn = np.asarray(results["bids_winners_dyn"], dtype=float)

        # Transfers / prices are REQUIRED for utility & revenue
        if "prices_winners_dyn" not in results:
            raise KeyError(
                "Missing 'prices_winners_dyn' in results. "
                "Log prices in evaluation simulation (CPC paid by winners) and return it."
            )
        prices_winners_dyn = np.asarray(results["prices_winners_dyn"], dtype=float)

        if "budgets_path_dyn" not in results:
            raise KeyError(
                "Missing 'budgets_eval_path' in results. "
                "Log full evaluation budgets with shape (T_eval, N_eval) (pre-spend budgets each period)."
            )
        budgets_eval_path = np.asarray(results["budgets_path_dyn"], dtype=float)

        if "bids_path_dyn" not in results:
            raise KeyError(
                "Missing 'bids_path_dyn' in results. "
                "Log full evaluation bids with shape (T_eval, N_eval) (bids each period for each player)."
            )
        bids_eval_path = np.asarray(results["bids_path_dyn"], dtype=float) 
        
        # Sanity checks for shapes
        if winners_idx_dyn.shape[0] != T_eval:
            raise ValueError(f"winners_idx_dyn has {winners_idx_dyn.shape[0]} rows but T_eval={T_eval}")
        if prices_winners_dyn.shape != winners_idx_dyn.shape:
            raise ValueError(f"prices_winners_dyn shape {prices_winners_dyn.shape} != winners_idx_dyn shape {winners_idx_dyn.shape}")
        if budgets_winners_dyn.shape != winners_idx_dyn.shape:
            raise ValueError(f"budgets_winners_dyn shape {budgets_winners_dyn.shape} != winners_idx_dyn shape {winners_idx_dyn.shape}")
        if bids_winners_dyn.shape != winners_idx_dyn.shape:
            raise ValueError(f"bids_winners_dyn shape {bids_winners_dyn.shape} != winners_idx_dyn shape {winners_idx_dyn.shape}")
        if budgets_eval_path.shape != (T_eval, N_eval):
            raise ValueError(f"budgets_eval_path must have shape (T_eval, N_eval)=({T_eval},{N_eval}), got {budgets_eval_path.shape}")

        # Calculate revenue
        period_revenue = np.zeros(T_eval, dtype=float)
        for t in range(T_eval):
            for k in range(cfg.K):
                period_revenue[t] += float(self.alpha[k]) * float(prices_winners_dyn[t, k])

        revenue_mean = float(np.mean(period_revenue)) if T_eval > 0 else np.nan
        revenue_std = float(np.std(period_revenue)) if T_eval > 0 else np.nan
        revenue_first_half = float(np.mean(period_revenue[:half])) if half > 0 else revenue_mean
        revenue_second_half = float(np.mean(period_revenue[half:])) if half > 0 else revenue_mean

        # calculate budget stats
        budget_mean_all = float(np.mean(budgets_eval_path)) if T_eval > 0 else np.nan
        budget_std_all = float(np.std(budgets_eval_path)) if T_eval > 0 else np.nan
        budget_min = float(np.min(budgets_eval_path)) if T_eval > 0 else np.nan
        budget_max = float(np.max(budgets_eval_path)) if T_eval > 0 else np.nan
        budget_mean_first_half = float(np.mean(budgets_eval_path[:half, :])) if half > 0 else budget_mean_all
        budget_mean_second_half = float(np.mean(budgets_eval_path[half:, :])) if half > 0 else budget_mean_all

        # Type groups: (iv, ibeta, idelta)
        type_counts = np.zeros((self.n_v, self.n_beta, self.n_delta), dtype=int)
        for pid in range(N_eval):
            iv = int(eval_v_idx[pid])
            ib = int(eval_beta_idx[pid])
            idl = int(eval_delta_idx[pid])
            type_counts[iv, ib, idl] += 1

        # Utility by type: alpha*(v-price)
        type_util_sum = np.zeros((self.n_v, self.n_beta, self.n_delta), dtype=float)
        for t in range(T_eval):
            for k in range(cfg.K):
                pid = int(winners_idx_dyn[t, k])
                iv = int(eval_v_idx[pid])
                ib = int(eval_beta_idx[pid])
                idl = int(eval_delta_idx[pid])
                v = float(self.v_set[iv])
                price = float(prices_winners_dyn[t, k])
                type_util_sum[iv, ib, idl] += float(self.alpha[k]) * (v - price)

        type_util_avg = type_util_sum / float(T_eval) if T_eval > 0 else type_util_sum

        type_util_per_player = np.zeros_like(type_util_avg)
        for iv in range(self.n_v):
            for ib in range(self.n_beta):
                for idl in range(self.n_delta):
                    n = int(type_counts[iv, ib, idl])
                    type_util_per_player[iv, ib, idl] = (type_util_avg[iv, ib, idl] / n) if n > 0 else 0.0

        # Compute mean budget for each player across time, then average within type
        player_budget_time_mean = np.mean(budgets_eval_path, axis=0) if T_eval > 0 else np.zeros(N_eval, dtype=float)
        type_budget_sum = np.zeros((self.n_v, self.n_beta, self.n_delta), dtype=float)
        for pid in range(N_eval):
            iv = int(eval_v_idx[pid])
            ib = int(eval_beta_idx[pid])
            idl = int(eval_delta_idx[pid])
            type_budget_sum[iv, ib, idl] += float(player_budget_time_mean[pid])

        type_budget_avg_per_player = np.zeros_like(type_budget_sum)
        for iv in range(self.n_v):
            for ib in range(self.n_beta):
                for idl in range(self.n_delta):
                    n = int(type_counts[iv, ib, idl])
                    type_budget_avg_per_player[iv, ib, idl] = (type_budget_sum[iv, ib, idl] / n) if n > 0 else 0.0

        # Average bid per player in evaluation
        player_bid_time_mean = np.mean(bids_eval_path, axis=0) if T_eval > 0 else np.zeros(N_eval, dtype=float)

        # Average rank per player in evaluation
        if T_eval > 0:
            ranks_eval_path = np.empty((T_eval, N_eval), dtype=int)
            for t in range(T_eval):
                noise = 1e-12 * self.rng.standard_normal(N_eval)
                order = np.argsort(-(bids_eval_path[t, :] + noise))
                ranks = np.empty(N_eval, dtype=int)
                ranks[order] = np.arange(1, N_eval + 1)
                ranks_eval_path[t, :] = ranks

            player_rank_time_mean = np.mean(ranks_eval_path, axis=0)
        else:
            player_rank_time_mean = np.zeros(N_eval, dtype=float)

        # ------------------------------------------------------------
        # WRITE FILE
        # ------------------------------------------------------------
        with open(out_path, "w", encoding="utf-8") as f:
            
            # header
            f.write("HEADER\n")
            kv = [
                ("N_est", cfg.N),
                ("K", cfg.K),
                ("alpha", ",".join([str(x) for x in cfg.alpha])),
                ("v_set", ",".join([str(x) for x in cfg.v_set])),
                ("beta_set", ",".join([str(x) for x in cfg.beta_set])),
                ("delta_set", ",".join([str(x) for x in cfg.delta_set])),
                ("B_min", cfg.B_min),
                ("B_max", cfg.B_max),
                ("b_min", cfg.b_min),
                ("b_max", cfg.b_max),
                ("budget_growth", cfg.budget_growth),
                ("income", cfg.income),
                ("S", cfg.S),
                ("sim_T", cfg.sim_T),
                ("mc_C", cfg.mc_C),
                ("burn_in", cfg.burn_in),
                ("T_eval", cfg.T_eval),
                ("outer_max_iter", cfg.outer_max_iter),
                ("tol_mu", cfg.tol_mu),
                ("mu_damping", cfg.mu_damping),
                ("eval_max_iter", cfg.eval_max_iter),
                ("tol_W", cfg.tol_W),
                ("tol_policy_max", cfg.tol_policy_max),
                ("tol_policy_mean", getattr(cfg, "tol_policy_mean", "NA")),
                ("rng_seed", cfg.rng_seed),
            ]
            key_w = max(len(k) for k, _ in kv)
            for k, v in kv:
                f.write(f"{k:<{key_w}}  {v}\n")
            f.write("\n")

            # welfare summary
            f.write("WELFARE_SUMMARY\n")
            ws = [
                ("welfare_vcg", f"{welfare_vcg:.6f}"),
                ("welfare_gsp_mean", f"{welfare_gsp_mean:.6f}"),
                ("welfare_gsp_std", f"{welfare_gsp_std:.6f}"),
                ("welfare_gap", f"{welfare_gap:.6f}"),
                ("welfare_ratio", f"{welfare_ratio:.6f}" if np.isfinite(welfare_ratio) else "nan"),
            ]
            key_w2 = max(len(k) for k, _ in ws)
            for k, v in ws:
                f.write(f"{k:<{key_w2}}  {v}\n")
            f.write("\n")


            # welfare stats
            f.write("AGGREGATE_WELFARE_STATS\n")
            aw = [
                ("mean_all", f"{welfare_gsp_mean:.6f}"),
                ("mean_first_half", f"{welfare_first_half:.6f}"),
                ("mean_second_half", f"{welfare_second_half:.6f}"),
                ("std_all", f"{welfare_gsp_std:.6f}"),
                ("min", f"{welfare_min:.6f}"),
                ("max", f"{welfare_max:.6f}"),
            ]
            key_w3 = max(len(k) for k, _ in aw)
            for k, v in aw:
                f.write(f"{k:<{key_w3}}  {v}\n")
            f.write("\n")

            # revenue
            f.write("REVENUE_SUMMARY\n")
            rs = [
                ("mean_all", f"{revenue_mean:.6f}"),
                ("mean_first_half", f"{revenue_first_half:.6f}"),
                ("mean_second_half", f"{revenue_second_half:.6f}"),
                ("std_all", f"{revenue_std:.6f}"),
            ]
            key_w4 = max(len(k) for k, _ in rs)
            for k, v in rs:
                f.write(f"{k:<{key_w4}}  {v}\n")
            f.write("\n")

            corr = np.corrcoef(welfare_dyn, period_revenue)[0, 1]
            f.write(str(corr))
            
            # budget
            f.write("BUDGET_SUMMARY\n")
            bs = [
                ("mean_all", f"{budget_mean_all:.6f}"),
                ("mean_first_half", f"{budget_mean_first_half:.6f}"),
                ("mean_second_half", f"{budget_mean_second_half:.6f}"),
                ("std_all", f"{budget_std_all:.6f}"),
                ("min", f"{budget_min:.6f}"),
                ("max", f"{budget_max:.6f}"),
            ]
            key_w5 = max(len(k) for k, _ in bs)
            for k, v in bs:
                f.write(f"{k:<{key_w5}}  {v}\n")
            f.write("\n")

            # utility
            f.write("UTILITY_BY_TYPE\n")
            headers = ["iv", "ibeta", "idelta", "value", "beta", "delta", "n_players", "avg_util", "util_per_player"]
            widths  = [4,   6,      7,       8,      6,     6,      10,         14,        16]
            aligns  = [">", ">",    ">",     ">",    ">",   ">",    ">",       ">",       ">"]
            f.write(self._fmt_row(headers, widths, aligns) + "\n")
            for iv in range(self.n_v):
                for ib in range(self.n_beta):
                    for idl in range(self.n_delta):
                        n_players = int(type_counts[iv, ib, idl])
                        row = [
                            iv,
                            ib,
                            idl,
                            f"{float(self.v_set[iv]):.2f}",
                            f"{float(self.beta_set[ib]):.2f}",
                            f"{float(self.delta_set[idl]):.2f}",
                            n_players,
                            f"{float(type_util_avg[iv, ib, idl]):.6f}",
                            f"{float(type_util_per_player[iv, ib, idl]):.6f}",
                        ]
                        f.write(self._fmt_row(row, widths, aligns) + "\n")
            f.write("\n")

            # Budget by type
            f.write("BUDGET_BY_TYPE\n")
            headers = ["iv", "ibeta", "idelta", "value", "beta", "delta", "n_players", "avg_budget"]
            widths  = [4,   6,      7,       8,      6,     6,      10,         14]
            aligns  = [">", ">",    ">",     ">",    ">",   ">",    ">",       ">"]
            f.write(self._fmt_row(headers, widths, aligns) + "\n")
            for iv in range(self.n_v):
                for ib in range(self.n_beta):
                    for idl in range(self.n_delta):
                        n_players = int(type_counts[iv, ib, idl])
                        row = [
                            iv,
                            ib,
                            idl,
                            f"{float(self.v_set[iv]):.2f}",
                            f"{float(self.beta_set[ib]):.2f}",
                            f"{float(self.delta_set[idl]):.2f}",
                            n_players,
                            f"{float(type_budget_avg_per_player[iv, ib, idl]):.6f}",
                        ]
                        f.write(self._fmt_row(row, widths, aligns) + "\n")
            f.write("\n")

            # average bid by player
            f.write("AVERAGE_BID_BY_PLAYER\n")
            headers = ["player", "iv", "value", "ibeta", "beta", "idelta", "delta", "avg_bid"]
            widths  = [8,        4,    8,       6,       6,      7,        6,      14]
            aligns  = [">",      ">",  ">",     ">",     ">",    ">",      ">",    ">"]
            f.write(self._fmt_row(headers, widths, aligns) + "\n")
            for pid in range(N_eval):
                iv = int(eval_v_idx[pid])
                ib = int(eval_beta_idx[pid])
                idl = int(eval_delta_idx[pid])
                row = [
                    pid,
                    iv,
                    f"{float(self.v_set[iv]):.2f}",
                    ib,
                    f"{float(self.beta_set[ib]):.2f}",
                    idl,
                    f"{float(self.delta_set[idl]):.2f}",
                    f"{float(player_bid_time_mean[pid]):.6f}",
                ]
                f.write(self._fmt_row(row, widths, aligns) + "\n")
            f.write("\n")

            # average rank by player
            f.write("AVERAGE_RANK_BY_PLAYER\n")
            headers = ["player", "iv", "value", "ibeta", "beta", "idelta", "delta", "avg_rank"]
            widths  = [8,        4,    8,       6,       6,      7,        6,      14]
            aligns  = [">",      ">",  ">",     ">",     ">",    ">",      ">",    ">"]
            f.write(self._fmt_row(headers, widths, aligns) + "\n")
            for pid in range(N_eval):
                iv = int(eval_v_idx[pid])
                ib = int(eval_beta_idx[pid])
                idl = int(eval_delta_idx[pid])
                row = [
                    pid,
                    iv,
                    f"{float(self.v_set[iv]):.2f}",
                    ib,
                    f"{float(self.beta_set[ib]):.2f}",
                    idl,
                    f"{float(self.delta_set[idl]):.2f}",
                    f"{float(player_rank_time_mean[pid]):.6f}",
                ]
                f.write(self._fmt_row(row, widths, aligns) + "\n")
            f.write("\n")

            # estimation players
            f.write("ESTIMATION_PLAYERS\n")
            headers = ["player", "iv", "value", "B0", "ibeta", "beta", "idelta", "delta"]
            widths  = [8,        4,    8,       8,    6,       6,      7,        6]
            aligns  = [">",      ">",  ">",     ">",  ">",     ">",    ">",      ">"]
            f.write(self._fmt_row(headers, widths, aligns) + "\n")
            for i in range(int(cfg.N)):
                iv = int(est_v_idx[i])
                ib = int(est_beta_idx[i])
                idl = int(est_delta_idx[i])
                row = [
                    i,
                    iv,
                    f"{float(self.v_set[iv]):.2f}",
                    f"{float(est_B0[i]):.2f}",
                    ib,
                    f"{float(self.beta_set[ib]):.2f}",
                    idl,
                    f"{float(self.delta_set[idl]):.2f}",
                ]
                f.write(self._fmt_row(row, widths, aligns) + "\n")
            f.write("\n")

            # evaluation players
            f.write("EVALUATION_PLAYERS\n")
            headers = ["player", "iv", "value", "B0", "ibeta", "beta", "idelta", "delta"]
            widths  = [8,        4,    8,       8,    6,       6,      7,        6]
            aligns  = [">",      ">",  ">",     ">",  ">",     ">",    ">",      ">"]
            f.write(self._fmt_row(headers, widths, aligns) + "\n")
            for i in range(N_eval):
                iv = int(eval_v_idx[i])
                ib = int(eval_beta_idx[i])
                idl = int(eval_delta_idx[i])
                row = [
                    i,
                    iv,
                    f"{float(self.v_set[iv]):.2f}",
                    f"{float(eval_B0[i]):.2f}",
                    ib,
                    f"{float(self.beta_set[ib]):.2f}",
                    idl,
                    f"{float(self.delta_set[idl]):.2f}",
                ]
                f.write(self._fmt_row(row, widths, aligns) + "\n")
            f.write("\n")

            # VCG allocation
            vcg_winners = np.asarray(results["vcg_winners_idx"], dtype=int)
            f.write("VCG_ALLOCATION\n")
            headers = ["slot", "player", "value", "B0"]
            widths  = [6,      8,        8,       8]
            aligns  = [">",    ">",      ">",     ">"]
            f.write(self._fmt_row(headers, widths, aligns) + "\n")
            for k, pid in enumerate(vcg_winners):
                row = [
                    k,
                    int(pid),
                    f"{float(self.v_set[int(eval_v_idx[pid])]):.2f}",
                    f"{float(eval_B0[pid]):.2f}",
                ]
                f.write(self._fmt_row(row, widths, aligns) + "\n")
            f.write("\n")

            # period welfare
            f.write("PERIOD_WELFARE\n")
            headers = ["t", "welfare_gsp"]
            widths  = [6,  14]
            aligns  = [">", ">"]
            f.write(self._fmt_row(headers, widths, aligns) + "\n")
            for t, w in enumerate(welfare_dyn):
                f.write(self._fmt_row([t, f"{float(w):.6f}"], widths, aligns) + "\n")
            f.write("\n")

            # period revenue
            f.write("PERIOD_REVENUE\n")
            headers = ["t", "revenue"]
            widths  = [6,  14]
            aligns  = [">", ">"]
            f.write(self._fmt_row(headers, widths, aligns) + "\n")
            for t, r in enumerate(period_revenue):
                f.write(self._fmt_row([t, f"{float(r):.6f}"], widths, aligns) + "\n")
            f.write("\n")

            # GSP allocation in evaluation
            f.write("GSP_ALLOCATION\n")

            budget_cols = [f"B_p{p}" for p in range(N_eval)]
            bid_cols    = [f"bid_p{p}" for p in range(N_eval)]

            headers = ["t", "slot", "player", "value", "beta", "delta", "budget", "bid", "price"] + budget_cols + bid_cols

            widths  = [6,   6,      8,        8,      6,      6,       12,        8,     8] + [10]*N_eval + [10]*N_eval
            aligns  = [">", ">",    ">",      ">",    ">",    ">",     ">",       ">",   ">"] + [">"]*N_eval + [">"]*N_eval

            f.write(self._fmt_row(headers, widths, aligns) + "\n")

            for t in range(T_eval):
                budget_panel_row = [f"{float(budgets_eval_path[t, p]):.2f}" for p in range(N_eval)]
                bid_panel_row    = [f"{float(bids_eval_path[t, p]):.2f}"    for p in range(N_eval)]

                for k in range(cfg.K):
                    pid = int(winners_idx_dyn[t, k])
                    iv = int(eval_v_idx[pid])
                    ib = int(eval_beta_idx[pid])
                    idl = int(eval_delta_idx[pid])

                    row = [
                        t,
                        k,
                        pid,
                        f"{float(self.v_set[iv]):.2f}",
                        f"{float(self.beta_set[ib]):.2f}",
                        f"{float(self.delta_set[idl]):.2f}",
                        f"{float(budgets_winners_dyn[t, k]):.2f}",
                        f"{float(bids_winners_dyn[t, k]):.2f}",
                        f"{float(prices_winners_dyn[t, k]):.2f}",
                    ] + budget_panel_row + bid_panel_row

                    f.write(self._fmt_row(row, widths, aligns) + "\n")

    def _save_outputs(self, results: Dict[str, Any]) -> None:
        cfg = self.cfg
        out_dir = self._run_output_path()

        if cfg.save_npz:
            npz_path = os.path.join(out_dir, "arrays.npz")

            np.savez(
                npz_path,
                # grids and sets
                B_grid=results["B_grid"],
                b_grid=results["b_grid"],
                v_set=np.asarray(cfg.v_set, dtype=float),
                beta_set=results["beta_set"],
                delta_set=results["delta_set"],
                # estimation players
                est_v_idx=results["est_v_idx"],
                est_B0=results["est_B0"],
                est_beta_idx=results["est_beta_idx"],
                est_delta_idx=results["est_delta_idx"],
                # evaluation players
                eval_v_idx=results["eval_v_idx"],
                eval_B0=results["eval_B0"],
                eval_beta_idx=results["eval_beta_idx"],
                eval_delta_idx=results["eval_delta_idx"],
                # solved objects
                W_aux=results["W_aux"],
                policy=results["policy"],
                # beliefs and histories
                mu_samples=results["mu_samples"],
                ks_history=np.asarray(results["ks_history"], dtype=float),
                # benchmarks + logs
                welfare_vcg=np.asarray([results["welfare_vcg"]], dtype=float),
                vcg_winners_idx=results["vcg_winners_idx"],
                winners_idx_dyn=results["winners_idx_dyn"],
                welfare_dyn=results["welfare_dyn"],
                bids_winners_dyn=results["bids_winners_dyn"],
                prices_winners_dyn=results["prices_winners_dyn"],
                budgets_winners_dyn=results["budgets_winners_dyn"],
                budgets_path_dyn=results["budgets_path_dyn"],
                allow_pickle=False
            )

        if cfg.save_txt:
            txt_path = os.path.join(out_dir, "run_summary.txt")
            self._write_run_summary_txt(results, txt_path)

            hist_path = os.path.join(out_dir, "budget_bid_history.dat")
            self._write_budget_bid_history_txt(results, hist_path)

            pol_path = os.path.join(out_dir, "policy_by_type.dat")
            self._write_policy_by_type_txt(pol_path)

    # ============================================================
    # Main solve
    # ============================================================

    def solve(self) -> Dict[str, Any]:
        """
        This is the main function of entrance.
            1. Set up estimation and evaluation players and initial cutoff samples
            2. Calculate VCG welfare (for comparison with the GSP)
            3. Do the main loop.
                1. Run the inner solve function to update policy and auxiliary function
                2. Run simulation to get new cutoff samples
                3. Calculate break statistics (KS) and break if suitable
            4. Run final simulation.
            5. Save output.
        """
        cfg = self.cfg

        # Initialize estimation players and mu
        self._init_estimation_players()
        self._initialize_mu()

        assert self.est_v_idx is not None
        assert self.est_B0 is not None
        assert self.est_beta_idx is not None
        assert self.est_delta_idx is not None
        assert self.mu_samples is not None and self.mu_samples.shape == (cfg.S, cfg.K)

        # VCG benchmark on evaluation population
        eval_B0, eval_v_idx, eval_beta_idx, eval_delta_idx = self._build_eval_players()
        vcg_winners_idx = self._vcg_allocation(eval_v_idx)
        welfare_vcg = float(np.sum(self.alpha * self.v_set[eval_v_idx[vcg_winners_idx]]))

        # Histories for diagnostics
        mu_history: List[np.ndarray] = [self.mu_samples.copy()]
        ks_history: List[float] = []
        inner_diag_history: List[Dict[str, Any]] = []
        budgets_end_history: List[np.ndarray] = []

        # Outer loop
        B_start = self.est_B0.copy()
        for m in range(cfg.outer_max_iter):
            print(f"OUTER {m} start")

            solve_inner_t = perf_counter()
            inner_diag = self._solve_inner()
            inner_diag_history.append(inner_diag)
            print(f"OUTER {m} finished solve_inner in {perf_counter() - solve_inner_t:.3f} seconds")
            print(f"   Max diff policy={inner_diag['diff_policy']:.6g}")
            print(f"   Mean diff policy={inner_diag['mean_diff_policy']:.6g}")
            print(f"   Max diff W={inner_diag['diff_W']:.6g}")
            print(f"   Evalulation iterations={inner_diag['eval_iters']}")

            # Simulate pooled cutoffs under current policy using estimation players
            sim_t = perf_counter()
            raw_cutoffs, B_end = self._simulate_pool_cutoffs_and_budgets(
                B_start, self.est_v_idx, self.est_beta_idx, self.est_delta_idx
            )
            mu_new = self._resample_mu_from_pool(raw_cutoffs)
            print(f"OUTER {m} finished mu update simulation in {perf_counter() - sim_t:.3f} sec")

            ks = ks_distance_cutoffs_marginal_max(self.mu_samples, mu_new)
            ks_history.append(ks)
            budgets_end_history.append(B_end.copy())
            print(f"KS {m} = {ks:.6g}")

            # Damped update
            lam = cfg.mu_damping
            replace = self.rng.random(cfg.S) < lam
            mu_updated = self.mu_samples.copy()
            mu_updated[replace] = mu_new[replace]
            self.mu_samples = mu_updated
            mu_history.append(self.mu_samples.copy())

            # Update budgets to get a better start next time
            B_start = B_end

            if (
                ks < cfg.tol_mu
                #and inner_diag["diff_policy_max"] < cfg.tol_policy_max
                and inner_diag["mean_diff_policy"] < cfg.tol_policy_mean
                and inner_diag["diff_W"] < cfg.tol_W
            ):
                print("Convergence achieved on mu, policy (max & mean), and W.")
                break

        print(self.policy.shape)
        print(self.W_aux.shape)
        print(np.mean(self.policy))
        print(np.min(self.policy))
        print(np.max(self.policy))

        
        # print some policy slices to check patterns
        for ib in range(self.n_beta):
            for idl in range(self.n_delta):
                for iv in range(self.n_v):
                    pol_slice = self.policy[ib, idl, iv, :]
                    W_slice = self.W_aux[ib,idl,iv,:]
                    print(f"\nSLICE (iv,ib,idl)=({iv},{ib},{idl})  "
                          f"min={pol_slice.min()} max={pol_slice.max()} mean={pol_slice.mean()} "
                          f"nonzero={np.count_nonzero(pol_slice)}")
                    for iB in range(self.n_B):
                        print(f"  iB={iB:3d}  b={pol_slice[iB]}   W={W_slice[iB]}")
        
        eval_logs = self._simulate_and_log_allocations(eval_B0, eval_v_idx, eval_beta_idx, eval_delta_idx)

        results: Dict[str, Any] = {
            "config": cfg,
            # grids/sets
            "B_grid": self.B_grid,
            "b_grid": self.b_grid,
            "v_set": self.v_set.copy(),
            "beta_set": self.beta_set.copy(),
            "delta_set": self.delta_set.copy(),
            # estimation players
            "est_v_idx": self.est_v_idx.copy(),
            "est_B0": self.est_B0.copy(),
            "est_beta_idx": self.est_beta_idx.copy(),
            "est_delta_idx": self.est_delta_idx.copy(),
            # evaluation players
            "eval_v_idx": eval_v_idx.copy(),
            "eval_B0": eval_B0.copy(),
            "eval_beta_idx": eval_beta_idx.copy(),
            "eval_delta_idx": eval_delta_idx.copy(),
            # solved objects
            "W_aux": self.W_aux.copy(),
            "policy": self.policy.copy(),
            # beliefs and histories
            "mu_samples": self.mu_samples.copy(),
            "mu_history": mu_history,
            "ks_history": ks_history,
            "inner_diag_history": inner_diag_history,
            "budgets_end_history": budgets_end_history,
            # VCG benchmark
            "vcg_winners_idx": vcg_winners_idx,
            "welfare_vcg": welfare_vcg,
            # evaluation logs
            **eval_logs,
        }

        # welfare by subgroups
        welfare_stats = self._compute_welfare_statistics(results)
        results["welfare_stats"] = welfare_stats

        # Save outputs
        self._save_outputs(results)
        return results


# ============================================================
# Entrace point
# ============================================================

if __name__ == "__main__":

    cfg = ModelConfig(
        N=200,
        K=3,
        alpha=(1.0,0.6,0.3),

        B_min=0, B_max=200,
        b_min=0, b_max=100,

        budget_growth=0.05,
        income=10.0,

        v_set=(80,120),
        beta_set=(1.0,),
        delta_set=(0.95,),

        eval_max_iter=500,
        tol_W=0.01,
        tol_policy_max=1.0,
        tol_policy_mean=0.05,

        outer_max_iter=100,
        tol_mu=0.05,
        mu_damping=0.4,
        S=500,
        sim_T=50,
        mc_C=1000,

        burn_in=500,
        T_eval=1000,

        # a player profile specifies (value_index,beta_index,delta_index,initial_budget_index)
        # note: since budget grid is on integers, we get budget_index = budget
        eval_players_idx=[
            
            (0, 0, 0, 100),
            (0, 0, 0, 100),

            (1, 0, 0, 100),
            (1, 0, 0, 100),

            (0, 0, 0, 100),
            (0, 0, 0, 100),

            (1, 0, 0, 100),
            (1, 0, 0, 100),

            (0, 0, 0, 100),
            (0, 0, 0, 100),

            (1, 0, 0, 100),
            (1, 0, 0, 100),

            (0, 0, 0, 100),
            (0, 0, 0, 100),

            (1, 0, 0, 100),
            (1, 0, 0, 100),
        ],

        output_dir="results",
        run_name="files",
        save_npz=True,
        save_txt=True,

        rng_seed=125343
    )

    solver = DynamicGSPSolver(cfg)
    _results = solver.solve()
