//! Renormalization-Learning Rust Library
//!
//! Wilsonian renormalization group as a model for agent skill acquisition.
//!
//! **Motivation:** Learning IS coarse-graining. Each epoch is an RG step.
//!   - Coarse-graining (block-averaging spins) = abstracting away microscopic details
//!   - Fixed points = perfected / mastered skills
//!   - Critical exponents = learning rate predictions
//!   - Universality classes = skills that learn the same way
//!
//! Mathematical framework:
//!   - Spin lattice: a 2D grid of `f64` values representing skill levels or
//!     Ising-type degrees of freedom
//!   - RG step: coarse-grain `b×b` blocks → rescale → observe
//!   - Beta function: `β(g) = (g′ − g) / ln(b)` — drives the flow toward fixed points
//!   - Correlation-length exponent ν: extracted from correlation lengths
//!     across RG steps
//!   - Learning curve: `p(t) = p_max − A·t^{−α}` where `α = 1/ν`

use ndarray::Array2;

// ============================================================================
// Lattice
// ============================================================================

/// A 2D lattice of `f64` values — the fundamental data structure.
///
/// The lattice can represent:
/// - **Spin values** in an Ising-like system (physics / RG interpretation)
/// - **Skill levels** `∈ [0, 1]` where `0` = unlearned, `1` = fully mastered
///
/// Data is stored in row-major order: `data[[y, x]]` accesses site `(x, y)`.
#[derive(Debug, Clone)]
pub struct Lattice {
    /// Linear dimension (L × L lattice).
    pub L: usize,
    /// Row-major array of size L×L.
    pub data: Array2<f64>,
}

impl Lattice {
    /// Create an L×L lattice initialized to zero.
    pub fn new(L: usize) -> Self {
        assert!(L > 0, "Lattice dimension must be positive");
        Lattice {
            L,
            data: Array2::zeros((L, L)),
        }
    }

    /// Fill with uniform random values in `[lo, hi)` using a deterministic seed.
    pub fn random(&mut self, lo: f64, hi: f64, seed: u64) {
        let mut state = seed ^ 0x9E3779B97F4A7C15;
        let range = hi - lo;
        let n = self.L * self.L;
        let slice = self.data.as_slice_mut().unwrap();
        for i in 0..n {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let r = (state as f64) * (1.0 / 18446744073709551616.0_f64);
            slice[i] = lo + range * r;
        }
    }

    /// Deep copy of the lattice.
    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Magnetization (order parameter): average value over all sites.
    pub fn magnetization(&self) -> f64 {
        self.data.mean().unwrap_or(0.0)
    }

    /// Energy: sum over all nearest-neighbor products.
    ///
    /// `H = −Σ_{⟨ij⟩} s_i · s_j`  (Ising-like interaction with periodic BC).
    pub fn energy(&self) -> f64 {
        let L = self.L;
        if L < 2 {
            return 0.0;
        }
        let mut e = 0.0;
        for y in 0..L {
            for x in 0..L {
                let s = self.data[[y, x]];
                // Right neighbour (periodic)
                let xr = (x + 1) % L;
                let sr = self.data[[y, xr]];
                e -= s * sr;
                // Bottom neighbour (periodic)
                let yb = (y + 1) % L;
                let sb = self.data[[yb, x]];
                e -= s * sb;
            }
        }
        e
    }

    /// Spatial correlation function `C(r)` for `0 ≤ r < L`.
    ///
    /// `C(r) = ⟨s_i · s_{i+r}⟩ − ⟨s_i⟩²` averaged over all sites
    /// in both horizontal and vertical directions for improved statistics.
    pub fn correlation(&self, r: usize) -> f64 {
        let L = self.L;
        if r >= L {
            return 0.0;
        }
        let mag = self.magnetization();
        let mut sum = 0.0;
        let mut count = 0;
        for y in 0..L {
            for x in 0..L {
                let si = self.data[[y, x]];
                // Horizontal
                let xr = (x + r) % L;
                let sj_h = self.data[[y, xr]];
                sum += si * sj_h;
                count += 1;
                // Vertical
                let yr = (y + r) % L;
                let sj_v = self.data[[yr, x]];
                sum += si * sj_v;
                count += 1;
            }
        }
        (sum / count as f64) - (mag * mag)
    }
}

// ============================================================================
// Coarse-Graining Operator
// ============================================================================

/// Block-spin transformation — the core coarse-graining operation.
///
/// Groups the lattice into `b×b` blocks and replaces each block with its
/// average value. This is the Wilsonian "integrating out" of short-distance
/// fluctuations, which in the learning analogy corresponds to abstracting
/// away microscopic details.
#[derive(Debug, Clone)]
pub struct CoarseGrainingOperator {
    /// Block factor.
    pub b: usize,
}

impl CoarseGrainingOperator {
    /// Create a new coarse-graining operator with block factor `b`.
    ///
    /// # Panics
    /// Panics if `b == 0`.
    pub fn new(b: usize) -> Self {
        assert!(b > 0, "Block factor must be positive");
        CoarseGrainingOperator { b }
    }

    /// Coarse-grain a lattice: group into `b×b` blocks, average each.
    /// Returns a new lattice of size `(L/b) × (L/b)`.
    ///
    /// # Panics
    /// Panics if `L % b != 0`.
    pub fn coarse_grain(&self, lat: &Lattice) -> Lattice {
        let L = lat.L;
        let b = self.b;
        assert!(
            L % b == 0,
            "Lattice dimension {} must be divisible by block factor {}",
            L,
            b
        );
        let newL = L / b;
        let mut result = Lattice::new(newL);
        for y in 0..newL {
            for x in 0..newL {
                let mut sum = 0.0;
                for dy in 0..b {
                    for dx in 0..b {
                        let sx = x * b + dx;
                        let sy = y * b + dy;
                        sum += lat.data[[sy, sx]];
                    }
                }
                result.data[[y, x]] = sum / (b * b) as f64;
            }
        }
        result
    }

    /// Rescale spin values by a multiplicative factor `zeta`.
    ///
    /// Under RG, spin fields rescale as `s' = s · ζ` where
    /// `ζ = b^{(2−d)/2}`. For `d=2`, `ζ = 1` (trivial at the Gaussian
    /// fixed point). Provided as a parameter for general use.
    pub fn rescale(&self, lat: &mut Lattice, zeta: f64) {
        lat.data.mapv_inplace(|v| v * zeta);
    }

    /// One full RG step: coarse-grain then rescale.
    pub fn rg_step(&self, lat: &Lattice, zeta: f64) -> Lattice {
        let mut coarse = self.coarse_grain(lat);
        self.rescale(&mut coarse, zeta);
        coarse
    }
}

// ============================================================================
// RG Transform
// ============================================================================

/// A multi-step RG flow: coarse-graining × scale × compute observables.
///
/// Stores the lattice at each RG level (decreasing resolution).
#[derive(Debug, Clone)]
pub struct RGTransform {
    /// Levels: `levels[0]` = original lattice.
    pub levels: Vec<Lattice>,
    /// Block factor used.
    pub b: usize,
    /// Rescale factor used.
    pub zeta: f64,
    /// Number of RG steps taken (= levels.len() - 1).
    pub nsteps: usize,
}

impl RGTransform {
    /// Run an RG flow for `nsteps` steps starting from `lat`.
    ///
    /// # Panics
    /// Panics if `nsteps` would reduce the lattice below 1×1.
    pub fn new(lat: &Lattice, b: usize, zeta: f64, nsteps: usize) -> Self {
        assert!(
            lat.L / b.pow(nsteps as u32) >= 1,
            "Too many RG steps: lattice would vanish"
        );
        let cg = CoarseGrainingOperator::new(b);
        let mut levels = Vec::with_capacity(nsteps + 1);
        levels.push(lat.clone());
        let mut current = lat.clone();
        for _ in 0..nsteps {
            current = cg.rg_step(&current, zeta);
            levels.push(current.clone());
        }
        RGTransform {
            levels,
            b,
            zeta,
            nsteps,
        }
    }

    /// Return the lattice at level `i` (0 = original).
    pub fn level(&self, i: usize) -> &Lattice {
        &self.levels[i]
    }

    /// Number of stored levels (original + nsteps).
    pub fn nlevels(&self) -> usize {
        self.levels.len()
    }

    /// Compute magnetization at each RG level.
    pub fn magnetization_flow(&self) -> Vec<f64> {
        self.levels.iter().map(|l| l.magnetization()).collect()
    }

    /// Compute energy at each RG level.
    pub fn energy_flow(&self) -> Vec<f64> {
        self.levels.iter().map(|l| l.energy()).collect()
    }

    /// Estimate correlation length at level `i` from the correlation function.
    ///
    /// Defined as the distance `r` where `C(r)` first drops below `C(0) / e`.
    pub fn correlation_length(&self, i: usize) -> f64 {
        let lat = &self.levels[i];
        let c0 = lat.correlation(0);
        if c0.abs() < 1e-15 {
            return 0.0;
        }
        let threshold = c0 / std::f64::consts::E;
        let L = lat.L;
        for r in 1..L {
            let cr = lat.correlation(r);
            if cr.abs() < threshold.abs() {
                return r as f64;
            }
        }
        L as f64
    }

    /// Correlation lengths across all RG levels.
    pub fn correlation_lengths(&self) -> Vec<f64> {
        (0..self.nlevels())
            .map(|i| self.correlation_length(i))
            .collect()
    }
}

// ============================================================================
// Fixed Point Detector
// ============================================================================

/// Detects RG fixed points by monitoring correlation-length divergence and
/// operator relevance.
///
/// In the RG framework:
/// - A **fixed point** is a configuration unchanged by further RG steps
/// - **Relevant operators** grow under RG (positive eigenvalue of the
///   linearised RG transformation)
/// - **Irrelevant operators** shrink under RG
/// - **Marginal operators** are unchanged (eigenvalue = 1)
///
/// For skill acquisition, a mastered skill corresponds to an RG fixed point.
#[derive(Debug)]
pub struct FixedPointDetector {
    /// Convergence tolerance for magnetization.
    pub tol: f64,
}

impl FixedPointDetector {
    /// Create a new fixed point detector.
    pub fn new(tol: f64) -> Self {
        FixedPointDetector { tol }
    }

    /// Find a fixed point by iterating RG until the magnetisation stabilises.
    ///
    /// Returns `(converged, steps_taken, final_magnetization)`.
    pub fn find_fixed_point(
        &self,
        lat0: &Lattice,
        b: usize,
        zeta: f64,
        max_steps: usize,
    ) -> (bool, usize, f64) {
        let cg = CoarseGrainingOperator::new(b);
        let mut lat = lat0.clone();
        let mut prev_mag = lat.magnetization();

        for steps in 0..max_steps {
            let next = cg.rg_step(&lat, zeta);
            lat = next;
            let mag = lat.magnetization();
            if (mag - prev_mag).abs() < self.tol {
                return (true, steps + 1, mag);
            }
            prev_mag = mag;
        }
        (false, max_steps, prev_mag)
    }

    /// Determine whether a skill lattice is an RG fixed point.
    ///
    /// A mastered skill is uniform with average ≥ `threshold`.
    pub fn skill_is_fixed_point(s: &Lattice, threshold: f64) -> bool {
        let mag = s.magnetization();
        if mag < threshold {
            return false;
        }
        for v in s.data.iter() {
            if (*v - mag).abs() > 1.0 - threshold {
                return false;
            }
        }
        true
    }

    /// Compute the beta function for a given coupling `g`.
    ///
    /// `β(g) = dg/dt` where `t = ln(b)`. At a fixed point, `β(g) = 0`.
    pub fn compute_beta_function(&self, g: f64, lat: &Lattice, b: usize, zeta: f64) -> f64 {
        let cg = CoarseGrainingOperator::new(b);
        let mut scaled = lat.clone();
        let cur_mag = lat.magnetization();
        let norm = if cur_mag.abs() < 1e-15 {
            1.0
        } else {
            g / cur_mag
        };
        scaled.data.mapv_inplace(|v| v * norm);

        let cg_lat = cg.rg_step(&scaled, zeta);
        let g_prime = cg_lat.magnetization();
        (g_prime - g) / (b as f64).ln()
    }
}

// ============================================================================
// Critical Exponent
// ============================================================================

/// Extracts critical exponents from RG flow data.
///
/// Key exponent: **ν** (correlation-length exponent).
///   - `ξ ~ |g − g_c|^{−ν}`  near criticality
///   - Fitted from `ξ_i ~ b^{i·ν}` across RG levels
///
/// Learning interpretation: `α = 1/ν` relates to the power-law learning rate.
#[derive(Debug)]
pub struct CriticalExponent {
    pub nu: f64,
}

impl CriticalExponent {
    /// Estimate ν from correlation lengths measured at successive RG levels.
    ///
    /// The fit is performed in log-space:
    ///   `ln(ξ_i) = ln(C) + i·ν·ln(b)`
    ///
    /// Returns `None` if fewer than 2 data points or fit fails.
    pub fn estimate_nu(lengths: &[f64], b: usize) -> Option<f64> {
        let n = lengths.len();
        if n < 2 || b < 2 {
            return None;
        }
        let lnb = (b as f64).ln();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;

        for (i, &len) in lengths.iter().enumerate() {
            if len <= 0.0 {
                return None;
            }
            let x = i as f64;
            let y = len.ln();
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let denom = n as f64 * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return None;
        }
        let slope = (n as f64 * sum_xy - sum_x * sum_y) / denom;
        let nu = slope / lnb;
        Some(nu.max(0.0))
    }

    /// Check whether two exponents define the same universality class within tolerance.
    pub fn same_universality_class(nu1: f64, nu2: f64, tol: f64) -> bool {
        (nu1 - nu2).abs() <= tol
    }

    /// The learning exponent `α = 1/ν`.
    pub fn learning_exponent(&self) -> f64 {
        1.0 / self.nu
    }
}

// ============================================================================
// Learning Rate Scheduler
// ============================================================================

/// An RG-inspired learning rate scheduler.
///
/// The core insight: the optimal learning rate at epoch `t` scales with the
/// RG beta function: `η(t) = η₀ · |β(g_t)|`.
///
/// Far from a fixed point, `|β(g)|` is large (big updates). Near a fixed point,
/// `|β(g)| → 0`, and the learning rate decays naturally.
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    /// Base learning rate.
    pub eta_0: f64,
    /// Block factor for RG steps.
    pub b: usize,
    /// Rescale factor.
    pub zeta: f64,
    /// Historical couplings (magnetizations) at each RG step.
    pub couplings: Vec<f64>,
    /// Learning rate at each step.
    pub history: Vec<f64>,
}

impl LearningRateScheduler {
    /// Create a new scheduler with base rate `eta_0`.
    pub fn new(eta_0: f64, b: usize, zeta: f64) -> Self {
        LearningRateScheduler {
            eta_0,
            b,
            zeta,
            couplings: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Compute the next learning rate based on the current lattice.
    pub fn step(&mut self, lat: &Lattice) -> f64 {
        let detector = FixedPointDetector::new(1e-10);
        let g = lat.magnetization();
        self.couplings.push(g);
        let beta = detector.compute_beta_function(g, lat, self.b, self.zeta);
        let lr = self.eta_0 * beta.abs().clamp(1e-10, 1.0);
        self.history.push(lr);
        lr
    }

    /// Estimate the learning-rate decay exponent `α_lr` from history.
    ///
    /// Fits `η(t) = η₀ · t^{−α_lr}` using the last `window` steps.
    pub fn decay_exponent(&self, window: usize) -> Option<f64> {
        let n = self.history.len().min(window);
        if n < 3 {
            return None;
        }
        let offset = self.history.len() - n;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;
        for i in 0..n {
            let t = (offset + i + 1) as f64;
            let eta = self.history[offset + i];
            if eta <= 0.0 {
                continue;
            }
            let x = t.ln();
            let y = eta.ln();
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }
        let denom = n as f64 * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 {
            return None;
        }
        let slope = (n as f64 * sum_xy - sum_x * sum_y) / denom;
        Some(-slope)
    }

    /// Reset the scheduler state.
    pub fn reset(&mut self) {
        self.couplings.clear();
        self.history.clear();
    }
}

// ============================================================================
// Learning Curve Analysis
// ============================================================================

/// Fit a power-law learning curve: `p(t) = p_max − A·t^{−α}`.
pub fn fit_power_law(t: &[f64], p: &[f64], p_max: f64) -> Option<PowerLawFit> {
    let n = t.len();
    if n < 3 || p_max <= 0.0 {
        return None;
    }
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let mut used = 0usize;

    for i in 0..n {
        let diff = p_max - p[i];
        if diff <= 1e-15 {
            continue;
        }
        let x = t[i].ln();
        let y = diff.ln();
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        used += 1;
    }

    if used < 2 {
        return None;
    }

    let denom = used as f64 * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return None;
    }

    let slope = (used as f64 * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / used as f64;

    let mut alpha = -slope;
    if alpha < 0.0 {
        alpha = 1e-6;
    }
    let a = intercept.exp();

    let mut ss = 0.0;
    let mut cnt = 0usize;
    for i in 0..n {
        let diff = p_max - p[i];
        if diff <= 1e-15 {
            continue;
        }
        let predicted = p_max - a * t[i].powf(-alpha);
        ss += (p[i] - predicted).powi(2);
        cnt += 1;
    }
    let rmse = if cnt > 0 { (ss / cnt as f64).sqrt() } else { 0.0 };

    Some(PowerLawFit { a, alpha, rmse, p_max })
}

/// Result of a power-law fit.
#[derive(Debug, Clone)]
pub struct PowerLawFit {
    pub a: f64,
    pub alpha: f64,
    pub rmse: f64,
    pub p_max: f64,
}

impl PowerLawFit {
    /// Predict the time to reach a target performance level.
    pub fn predict_mastery_time(&self, target: f64) -> Option<f64> {
        if target >= self.p_max || self.a <= 0.0 || self.alpha <= 0.0 {
            return None;
        }
        let diff = self.p_max - target;
        if diff <= 0.0 {
            return None;
        }
        Some((self.a / diff).powf(1.0 / self.alpha))
    }

    /// Compare two learning curves.
    /// Returns `|α₁ − α₂| / tol`. Value < 1.0 means same universality class.
    pub fn compare(&self, other: &PowerLawFit, tol: f64) -> f64 {
        (self.alpha - other.alpha).abs() / tol
    }
}

mod tests;