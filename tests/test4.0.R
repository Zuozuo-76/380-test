# Unit tests for the STA380 k-NN Monte Carlo functions
# These tests are intended to be run with testthat.
#
# In a package, these would typically live under:
#   tests/testthat/test-knn-mc.R
#
# For the Basic Implementation Check, a single test file is acceptable.

library(testthat)

# If developing as a package, you would use:
# devtools::load_all()
# Here, source the functions directly:
source("function4.0.R")

test_that("dgp_x returns correct dimensions", {
  x1 <- dgp_x(n = 10, d = 1, seed = 1)
  x2 <- dgp_x(n = 10, d = 2, seed = 1)

  expect_equal(dim(x1), c(10, 1))
  expect_equal(dim(x2), c(10, 2))
})

test_that("dgp_f matches known special values for 1d", {
  expect_equal(dgp_f(0, type = "1d"), 0, tolerance = 1e-12)
  expect_equal(dgp_f(0.5, type = "1d"), 0, tolerance = 1e-12)
})

test_that("knn_predict with k = n equals the training mean", {
  x_train <- matrix(c(0, 0.5, 1), ncol = 1)
  y_train <- c(1, 2, 3)
  x_test <- c(0.2, 0.8)

  pred <- knn_predict(x_train, y_train, x_test, k = length(y_train))

  expect_equal(pred, rep(mean(y_train), length(x_test)))
})

test_that("simulate_one_fit returns expected structure", {
  res <- simulate_one_fit(
    n = 30, k = 3, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 123
  )

  expect_true(is.list(res))
  expect_true(all(c("settings", "grid_axis", "fit", "train") %in% names(res)))
  expect_equal(nrow(res$fit), 21)
  expect_equal(nrow(res$train), 30)
  expect_true(all(c("x1", "f_true", "f_hat") %in% names(res$fit)))
})

test_that("simulate_one_fit is reproducible with the same seed", {
  a <- simulate_one_fit(
    n = 30, k = 3, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 123
  )

  b <- simulate_one_fit(
    n = 30, k = 3, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 123
  )

  expect_equal(a$fit, b$fit)
  expect_equal(a$train, b$train)
})

test_that("mc_knn_decomp returns expected structure and valid components", {
  res <- mc_knn_decomp(
    n = 30, k = 3, B = 40, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 123
  )

  expect_true(is.list(res))
  expect_true(all(c("settings", "grid_axis", "pointwise") %in% names(res)))

  pw <- res$pointwise
  expect_equal(nrow(pw), 21)
  expect_true(all(pw$variance >= 0))
  expect_true(all(pw$bias2 >= 0))
  expect_equal(pw$mse_f, pw$bias2 + pw$variance, tolerance = 1e-6)
  expect_equal(pw$mse_y, pw$mse_f + res$settings$sigma^2, tolerance = 1e-6)
})

test_that("mc_knn_curve returns one row per k", {
  curve <- mc_knn_curve(
    k_values = c(1, 3, 5),
    n = 30, B = 20, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 1
  )

  expect_equal(nrow(curve), 3)
  expect_equal(curve$k, c(1, 3, 5))
})

test_that("mc_knn_curve_components returns expected columns and nonnegative values", {
  curve2 <- mc_knn_curve_components(
    k_values = c(1, 3, 5),
    n = 30, B = 20, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 1
  )

  expect_s3_class(curve2, "data.frame")
  expect_equal(nrow(curve2), 3)

  expect_true(all(c(
    "k", "avg_bias2", "avg_variance", "avg_mse_f", "avg_mse_y"
  ) %in% names(curve2)))

  expect_true(all(curve2$avg_bias2 >= 0))
  expect_true(all(curve2$avg_variance >= 0))
  expect_true(all(curve2$avg_mse_f >= 0))
  expect_true(all(curve2$avg_mse_y >= curve2$avg_mse_f))
})

test_that("mc_knn_curve_components is reproducible with the same seed", {
  curve_a <- mc_knn_curve_components(
    k_values = c(1, 3, 5),
    n = 30, B = 20, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 123
  )

  curve_b <- mc_knn_curve_components(
    k_values = c(1, 3, 5),
    n = 30, B = 20, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 123
  )

  expect_equal(curve_a, curve_b)
})

test_that("mc_knn_compare_curve returns expected columns", {
  cmp <- mc_knn_compare_curve(
    k_values = c(1, 3, 5),
    n = 30, B = 20, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 1
  )

  expect_s3_class(cmp, "data.frame")
  expect_equal(nrow(cmp), 3)

  expect_true(all(c(
    "k", "monte_carlo_mse", "theoretical_mse"
  ) %in% names(cmp)))

  expect_true(all(cmp$monte_carlo_mse >= 0))
  expect_true(all(cmp$theoretical_mse >= 0))
})

testthat::skip_if_not_installed("ggplot2")
testthat::skip_if_not_installed("tidyr")

test_that("plot_fit_vs_truth_1d returns a ggplot object", {
  res <- simulate_one_fit(
    n = 30, k = 3, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 123
  )

  p <- plot_fit_vs_truth_1d(res)
  expect_true(inherits(p, "ggplot"))
})

test_that("plot_mse_1d returns a ggplot object", {
  res <- mc_knn_decomp(
    n = 30, k = 3, B = 20, sigma = 0.2,
    d = 1, type = "1d", grid_size = 21, seed = 123
  )

  p <- plot_mse_1d(res)
  expect_true(inherits(p, "ggplot"))
})

test_that("plot_curve_components returns a ggplot object", {
  curve2 <- data.frame(
    k = c(1, 3, 5),
    avg_bias2 = c(0.10, 0.20, 0.30),
    avg_variance = c(0.40, 0.30, 0.20),
    avg_mse_f = c(0.50, 0.50, 0.50),
    avg_mse_y = c(0.54, 0.54, 0.54)
  )

  p <- plot_curve_components(
    curve2,
    show = c("avg_bias2", "avg_variance", "avg_mse_y")
  )

  expect_true(inherits(p, "ggplot"))
})

test_that("plot_mse_compare_curve returns a ggplot object", {
  cmp <- data.frame(
    k = c(1, 3, 5),
    monte_carlo_mse = c(0.60, 0.50, 0.65),
    theoretical_mse = c(0.58, 0.49, 0.63)
  )

  p <- plot_mse_compare_curve(cmp)
  expect_true(inherits(p, "ggplot"))
})
