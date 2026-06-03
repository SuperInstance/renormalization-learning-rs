//! Test suite for the Renormalization-Learning library.
//!
//! Tests cover: lattice operations, block spin RG, fixed points,
//! critical exponents, learning rate scheduling, learning curve analysis,
//! and edge cases.

use crate::{Lattice, CoarseGrainingOperator, RGTransform, FixedPointDetector, CriticalExponent, LearningRateScheduler, PowerLawFit, fit_power_law};

// ─── Lattice tests ─────────────────────────────────────────────────

#[test]
fn test_lattice_create() {
    let lat = Lattice::new(8);
    assert_eq!(lat.L, 8);
    assert_eq!(lat.data.len(), 64);
    for v in lat.data.iter() {
        assert!((*v).abs() < 1e-15);
    }
}

#[test]
fn test_magnetization_uniform() {
    let mut lat = Lattice::new(4);
    lat.data.fill(1.0);
    assert!((lat.magnetization() - 1.0).abs() < 1e-15);
    lat.data.fill(-0.5);
    assert!((lat.magnetization() - (-0.5)).abs() < 1e-15);
}

#[test]
fn test_energy_ferromagnetic() {
    let L = 8;
    let mut lat = Lattice::new(L);
    lat.data.fill(1.0);
    // 2D square lattice with periodic BC: each site has 4 neighbors.
    // Each bond counted once: total bonds = L^2 * 2 = 128.
    // Energy = -128.
    let e = lat.energy();
    assert!((e - (-128.0)).abs() < 1e-10);
}

#[test]
fn test_energy_antiferro() {
    let L = 4;
    let mut lat = Lattice::new(L);
    for y in 0..L {
        for x in 0..L {
            lat.data[[y, x]] = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    let e = lat.energy();
    // Checkerboard: each +1 has four -1 neighbors => product = -1,
    // so -sum = +1 per bond. Total bonds = L^2 * 2 = 32. Energy = 32.
    assert!((e - 32.0).abs() < 1e-10);
}

#[test]
fn test_correlation_zero() {
    let mut lat = Lattice::new(4);
    for i in 0..16 {
        lat.data.as_slice_mut().unwrap()[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    let c0 = lat.correlation(0);
    // C(0) = <s_i^2> - <s_i>^2 = 1 - 0 = 1
    assert!((c0 - 1.0).abs() < 1e-10);
}

#[test]
fn test_correlation_uniform() {
    let mut lat = Lattice::new(4);
    lat.data.fill(1.0);
    let c = lat.correlation(1);
    assert!((c).abs() < 1e-10);
}

#[test]
fn test_lattice_copy() {
    let mut lat = Lattice::new(6);
    lat.random(-1.0, 1.0, 42);
    let cpy = lat.copy();
    assert_eq!(cpy.L, lat.L);
    assert!((cpy.data - &lat.data).mapv(|v| v.abs()).sum() < 1e-15);
}

#[test]
fn test_single_site() {
    let mut lat = Lattice::new(1);
    lat.data[[0, 0]] = 1.0;
    assert!((lat.magnetization() - 1.0).abs() < 1e-15);
    assert!((lat.energy() - 0.0).abs() < 1e-15);
    let c0 = lat.correlation(0);
    // C(0) = <s_i^2> - <s_i>^2 = 1 - 1 = 0
    assert!((c0).abs() < 1e-15);
}

// ─── Coarse-Graining tests ─────────────────────────────────────────

#[test]
fn test_block_spin_reduces() {
    let mut lat = Lattice::new(8);
    lat.random(-1.0, 1.0, 123);
    let op = CoarseGrainingOperator::new(2);
    let cg = op.coarse_grain(&lat);
    assert_eq!(cg.L, 4);
}

#[test]
fn test_block_spin_reduces4() {
    let mut lat = Lattice::new(16);
    lat.random(-1.0, 1.0, 456);
    let op = CoarseGrainingOperator::new(4);
    let cg = op.coarse_grain(&lat);
    assert_eq!(cg.L, 4);
}

#[test]
fn test_mag_preserved_uniform() {
    let mut lat = Lattice::new(8);
    lat.data.fill(0.75);
    let mag0 = lat.magnetization();
    let op = CoarseGrainingOperator::new(2);
    let cg = op.coarse_grain(&lat);
    let mag_cg = cg.magnetization();
    assert!((mag_cg - mag0).abs() < 1e-15);
}

#[test]
fn test_rg_b1() {
    let mut lat = Lattice::new(4);
    lat.random(-1.0, 1.0, 77);
    let op = CoarseGrainingOperator::new(1);
    let cg = op.coarse_grain(&lat);
    assert_eq!(cg.L, 4);
    let mag0 = lat.magnetization();
    let mag1 = cg.magnetization();
    assert!((mag1 - mag0).abs() < 1e-15);
}

#[test]
fn test_rg_step_reduces() {
    let mut lat = Lattice::new(16);
    lat.random(-1.0, 1.0, 42);
    let op = CoarseGrainingOperator::new(2);
    let cg = op.rg_step(&lat, 1.0);
    assert_eq!(cg.L, 8);
}

#[test]
fn test_rescale_factor() {
    let mut lat = Lattice::new(4);
    lat.data.fill(1.0);
    let op = CoarseGrainingOperator::new(2);
    op.rescale(&mut lat, 2.0);
    assert!((lat.data[[0, 0]] - 2.0).abs() < 1e-15);
}

// ─── RG Transform tests ────────────────────────────────────────────

#[test]
fn test_rgflow_create() {
    let mut lat = Lattice::new(16);
    lat.random(-1.0, 1.0, 42);
    let flow = RGTransform::new(&lat, 2, 1.0, 3);
    assert_eq!(flow.nlevels(), 4);
    assert_eq!(flow.level(0).L, 16);
    assert_eq!(flow.level(1).L, 8);
    assert_eq!(flow.level(2).L, 4);
    assert_eq!(flow.level(3).L, 2);
}

#[test]
fn test_rgflow_magnetization_flow() {
    let mut lat = Lattice::new(8);
    lat.random(-1.0, 1.0, 42);
    let flow = RGTransform::new(&lat, 2, 1.0, 2);
    let mags = flow.magnetization_flow();
    assert_eq!(mags.len(), 3);
    for &m in &mags {
        assert!(m.is_finite());
    }
}

#[test]
fn test_correlation_length_uniform() {
    let mut lat = Lattice::new(8);
    lat.data.fill(1.0);
    let flow = RGTransform::new(&lat, 2, 1.0, 2);
    let cl = flow.correlation_length(0);
    assert!(cl < 1.0);
}

#[test]
fn test_correlation_lengths_length() {
    let mut lat = Lattice::new(16);
    lat.random(-1.0, 1.0, 99);
    let flow = RGTransform::new(&lat, 2, 1.0, 3);
    let cls = flow.correlation_lengths();
    assert_eq!(cls.len(), 4);
}

#[test]
fn test_rgflow_energy_flow() {
    let mut lat = Lattice::new(8);
    lat.random(-1.0, 1.0, 42);
    let flow = RGTransform::new(&lat, 2, 1.0, 2);
    let energies = flow.energy_flow();
    assert_eq!(energies.len(), 3);
    for &e in &energies {
        assert!(e.is_finite());
    }
}

// ─── Fixed Point Detector tests ────────────────────────────────────

#[test]
fn test_fixed_point_uniform() {
    let mut lat = Lattice::new(8);
    lat.data.fill(1.0);
    let detector = FixedPointDetector::new(1e-10);
    let (converged, steps, mag) = detector.find_fixed_point(&lat, 2, 1.0, 20);
    assert!(converged);
    assert!(steps > 0);
    assert!((mag - 1.0).abs() < 1e-10);
}

#[test]
fn test_fixed_point_nearly_uniform() {
    let mut lat = Lattice::new(8);
    lat.data.fill(0.999);
    lat.data[[0, 0]] += 0.001;
    let detector = FixedPointDetector::new(1e-6);
    let (converged, _steps, mag) = detector.find_fixed_point(&lat, 2, 1.0, 20);
    assert!(converged);
    assert!((mag - 1.0).abs() < 0.01);
}

#[test]
fn test_beta_function() {
    let mut lat = Lattice::new(8);
    lat.random(-1.0, 1.0, 99);
    let detector = FixedPointDetector::new(1e-10);
    let beta = detector.compute_beta_function(1.0, &lat, 2, 1.0);
    assert!(beta.is_finite());
}

#[test]
fn test_beta_function_uniform() {
    let mut lat = Lattice::new(8);
    lat.data.fill(1.0);
    let detector = FixedPointDetector::new(1e-10);
    let beta = detector.compute_beta_function(1.0, &lat, 2, 1.0);
    assert!(beta.abs() < 0.01);
}

#[test]
fn test_skill_fixed_point_mastered() {
    let mut lat = Lattice::new(8);
    lat.data.fill(1.0);
    assert!(FixedPointDetector::skill_is_fixed_point(&lat, 0.99));
}

#[test]
fn test_skill_not_fixed() {
    let mut lat = Lattice::new(8);
    for y in 0..8 {
        for x in 0..8 {
            lat.data[[y, x]] = if (x + y) % 2 == 0 { 0.8 } else { 0.6 };
        }
    }
    assert!(!FixedPointDetector::skill_is_fixed_point(&lat, 0.99));
}

#[test]
fn test_skill_coarse_grain_preserves() {
    let mut lat = Lattice::new(16);
    lat.data.fill(1.0);
    let op = CoarseGrainingOperator::new(2);
    let cg = op.coarse_grain(&lat);
    let mag_cg = cg.magnetization();
    assert!((mag_cg - 1.0).abs() < 1e-15);
    assert!(FixedPointDetector::skill_is_fixed_point(&cg, 0.99));
}

// ─── Critical Exponent tests ───────────────────────────────────────

#[test]
fn test_critical_exponent_known() {
    let b = 2;
    let nu_true = 1.0;
    let lengths: Vec<f64> = (0..4).map(|i| (b as f64).powi(i) as f64).collect();
    let nu = CriticalExponent::estimate_nu(&lengths, b).unwrap();
    assert!((nu - nu_true).abs() < 0.01);
}

#[test]
fn test_critical_exponent_known_15() {
    let b = 2;
    let nu_true = 1.5;
    let lengths: Vec<f64> = (0..5).map(|i| (b as f64).powf(nu_true * i as f64)).collect();
    let nu = CriticalExponent::estimate_nu(&lengths, b).unwrap();
    assert!((nu - nu_true).abs() < 0.01);
}

#[test]
fn test_critical_exponent_few_points() {
    let lengths = vec![5.0]; // only 1 point
    assert!(CriticalExponent::estimate_nu(&lengths, 2).is_none());
}

#[test]
fn test_universality_class() {
    assert!(CriticalExponent::same_universality_class(1.0, 1.01, 0.05));
    assert!(!CriticalExponent::same_universality_class(1.0, 2.0, 0.1));
}

#[test]
fn test_learning_exponent() {
    let ce = CriticalExponent { nu: 2.0 };
    assert!((ce.learning_exponent() - 0.5).abs() < 1e-15);
}

#[test]
fn test_two_lattices_same_exponent() {
    let b = 2;
    let mut rng_state = 12345u64;
    let noise = |state: &mut u64| -> f64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        (*state as f64) * (1.0 / 18446744073709551616.0_f64)
    };
    let len1: Vec<f64> = (0..4).map(|i| {
        (b as f64).powf(1.5 * i as f64) * (1.0 + 0.01 * (noise(&mut rng_state) - 0.5) * 2.0)
    }).collect();
    let len2: Vec<f64> = (0..4).map(|i| {
        (b as f64).powf(1.5 * i as f64) * (1.0 + 0.01 * (noise(&mut rng_state) - 0.5) * 2.0)
    }).collect();
    let nu1 = CriticalExponent::estimate_nu(&len1, b).unwrap();
    let nu2 = CriticalExponent::estimate_nu(&len2, b).unwrap();
    assert!(CriticalExponent::same_universality_class(nu1, nu2, 0.2));
}

// ─── Learning Rate Scheduler tests ─────────────────────────────────

#[test]
fn test_lr_scheduler_step() {
    let mut lat = Lattice::new(8);
    lat.random(-1.0, 1.0, 42);
    let mut sched = LearningRateScheduler::new(0.1, 2, 1.0);
    let lr = sched.step(&lat);
    assert!(lr > 0.0);
    assert!(lr <= 0.1);
    assert_eq!(sched.history.len(), 1);
    assert_eq!(sched.couplings.len(), 1);
}

#[test]
fn test_lr_scheduler_multiple_steps() {
    let mut lat = Lattice::new(8);
    lat.random(-1.0, 1.0, 42);
    let mut sched = LearningRateScheduler::new(0.1, 2, 1.0);
    for _ in 0..5 {
        let lr = sched.step(&lat);
        assert!(lr > 0.0);
    }
    assert_eq!(sched.history.len(), 5);
}

#[test]
fn test_lr_scheduler_reset() {
    let mut lat = Lattice::new(8);
    lat.random(-1.0, 1.0, 42);
    let mut sched = LearningRateScheduler::new(0.1, 2, 1.0);
    sched.step(&lat);
    sched.step(&lat);
    sched.reset();
    assert!(sched.history.is_empty());
    assert!(sched.couplings.is_empty());
}

#[test]
fn test_lr_scheduler_decay_exponent_too_few() {
    let mut sched = LearningRateScheduler::new(0.1, 2, 1.0);
    assert!(sched.decay_exponent(10).is_none());
}

#[test]
fn test_lr_scheduler_decay_exponent() {
    let mut lat = Lattice::new(4);
    let mut sched = LearningRateScheduler::new(0.1, 2, 1.0);
    for _ in 0..10 {
        lat.random(-1.0, 1.0, 42);
        sched.step(&lat);
    }
    let alpha = sched.decay_exponent(10);
    // Should produce a finite exponent (may be near zero for random data)
    assert!(alpha.is_some());
    assert!(alpha.unwrap().is_finite());
}

// ─── Learning Curve Analysis tests ────────────────────────────────

#[test]
fn test_power_law_fit() {
    let p_max = 0.95;
    let a_true = 0.5;
    let alpha_true = 0.4;
    let n = 15;
    let t: Vec<f64> = (0..n).map(|i| (i * 10 + 5) as f64).collect();
    let p: Vec<f64> = t
        .iter()
        .map(|&ti| p_max - a_true * ti.powf(-alpha_true))
        .collect();

    let fit = fit_power_law(&t, &p, p_max).unwrap();
    assert!((fit.a - a_true).abs() < 0.01);
    assert!((fit.alpha - alpha_true).abs() < 0.01);
}

#[test]
fn test_predict_mastery_time() {
    let fit = PowerLawFit {
        a: 2.0,
        alpha: 0.5,
        rmse: 0.0,
        p_max: 1.0,
    };
    let t_pred = fit.predict_mastery_time(0.9).unwrap();
    // t = (2.0 / 0.1)^2 = 400
    assert!((t_pred - 400.0).abs() < 1.0);
}

#[test]
fn test_mastery_time_invalid_target_above_max() {
    let fit = PowerLawFit {
        a: 2.0,
        alpha: 0.5,
        rmse: 0.0,
        p_max: 1.0,
    };
    assert!(fit.predict_mastery_time(1.5).is_none());
}

#[test]
fn test_mastery_time_invalid_target_equal_max() {
    let fit = PowerLawFit {
        a: 2.0,
        alpha: 0.5,
        rmse: 0.0,
        p_max: 1.0,
    };
    assert!(fit.predict_mastery_time(1.0).is_none());
}

#[test]
fn test_compare_curves() {
    let tol = 0.1;
    let fit1 = PowerLawFit {
        a: 1.0,
        alpha: 0.5,
        rmse: 0.0,
        p_max: 1.0,
    };
    let fit2 = PowerLawFit {
        a: 1.0,
        alpha: 0.52,
        rmse: 0.0,
        p_max: 1.0,
    };
    assert!(fit1.compare(&fit2, tol) < 1.0);

    let fit3 = PowerLawFit {
        a: 1.0,
        alpha: 0.8,
        rmse: 0.0,
        p_max: 1.0,
    };
    assert!(fit1.compare(&fit3, tol) > 1.0);

    let fit4 = PowerLawFit {
        a: 1.0,
        alpha: 0.5,
        rmse: 0.0,
        p_max: 1.0,
    };
    assert!((fit1.compare(&fit4, 1.0)).abs() < 1e-15);
}

#[test]
fn test_fit_power_law_too_few_points() {
    let t = vec![1.0, 2.0];
    let p = vec![0.5, 0.7];
    assert!(fit_power_law(&t, &p, 1.0).is_none());
}

#[test]
fn test_fit_power_law_saturated() {
    let t = vec![1.0, 2.0, 3.0];
    let p = vec![1.0, 1.0, 1.0]; // all saturated at p_max
    let fit = fit_power_law(&t, &p, 1.0);
    assert!(fit.is_none());
}

#[test]
fn test_rmse_computation() {
    let p_max = 1.0;
    let t: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let p: Vec<f64> = t
        .iter()
        .map(|&ti| p_max - 0.5 * ti.powf(-0.3))
        .collect();
    let fit = fit_power_law(&t, &p, p_max).unwrap();
    assert!(fit.rmse > 0.0);
    assert!(fit.rmse < 0.01); // fit should be excellent on synthetic data
}

#[test]
fn test_phase_transition_magnetization() {
    let mut lat = Lattice::new(4);
    for i in 0..8 {
        lat.data.as_slice_mut().unwrap()[i] = 0.8;
    }
    for i in 8..16 {
        lat.data.as_slice_mut().unwrap()[i] = 0.2;
    }
    let mag = lat.magnetization();
    assert!((mag - 0.5).abs() < 1e-15);
}

// ─── Edge cases ────────────────────────────────────────────────────

#[test]
fn test_l2_correlation_index_bounds() {
    let lat = Lattice::new(8);
    // r < L should return valid numbers
    let c = lat.correlation(7);
    assert!(c.is_finite());
    // r >= L should return 0
    let c_out = lat.correlation(8);
    assert!((c_out).abs() < 1e-15);
}

#[test]
fn test_energy_for_small_lattice() {
    let lat = Lattice::new(1);
    assert!((lat.energy() - 0.0).abs() < 1e-15);
}

#[test]
fn test_random_values_in_range() {
    let mut lat = Lattice::new(100);
    lat.random(-5.0, 5.0, 1234);
    for v in lat.data.iter() {
        assert!(*v >= -5.0 && *v < 5.0);
    }
}

#[test]
fn test_different_random_seeds() {
    let mut lat1 = Lattice::new(4);
    lat1.random(0.0, 1.0, 42);
    let mut lat2 = Lattice::new(4);
    lat2.random(0.0, 1.0, 42);
    // Same seed => same values
    assert!((&lat1.data - &lat2.data).mapv(|v| v.abs()).sum() < 1e-15);

    let mut lat3 = Lattice::new(4);
    lat3.random(0.0, 1.0, 99);
    // Different seed => different values (overwhelmingly likely)
    let diff = (&lat1.data - &lat3.data).mapv(|v| v.abs()).sum();
    assert!(diff > 1e-10);
}

#[test]
fn test_rgflow_metadata() {
    let mut lat = Lattice::new(27);
    lat.random(-1.0, 1.0, 7);
    let flow = RGTransform::new(&lat, 3, 0.5, 2);
    assert_eq!(flow.b, 3);
    assert!((flow.zeta - 0.5).abs() < 1e-15);
    assert_eq!(flow.nsteps, 2);
}

#[test]
fn test_correlation_symmetric() {
    let mut lat = Lattice::new(8);
    lat.random(-1.0, 1.0, 42);
    let c1 = lat.correlation(2);
    let c2 = lat.correlation(6);
    // For an L=8 lattice, r=2 and r=6 are equivalent by periodicity
    assert!((c1 - c2).abs() < 1e-10);
}
