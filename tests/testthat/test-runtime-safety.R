test_that("fatal paths surface as R errors", {
  expect_error(llamacppR:::cpp_llamacpp_test_trigger_fatal(), "llama.cpp fatal test")
})
