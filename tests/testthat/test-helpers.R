test_that("gguf magic detection works", {
  path <- tempfile(fileext = ".gguf")
  writeBin(charToRaw("GGUFdummy"), path)
  expect_true(llamacpp_is_gguf(path))
})

test_that("gguf detection rejects missing files", {
  expect_false(llamacpp_is_gguf(tempfile(fileext = ".gguf")))
})

test_that("default model path is under the package cache", {
  path <- llamacppR:::llamacpp_default_model_path()
  expect_true(grepl("llamacppR", path, fixed = TRUE))
  expect_true(grepl("\\.gguf$", path))
})

test_that("default model presets are available", {
  expect_match(llamacpp_model_path("3b"), "3B-Instruct", ignore.case = FALSE)
  expect_match(llamacpp_model_path("0.5b"), "0.5B-Instruct", ignore.case = FALSE)
  expect_match(llamacpp_model_path("starcoder"), "starcoder2-3b-instruct", ignore.case = FALSE)
  expect_match(llamacpp_model_path("deepseek"), "DeepSeek-R1-Distill-Qwen-7B", ignore.case = FALSE)
})

test_that("list models finds gguf files and marks curated presets", {
  cache <- tempfile()
  dir.create(cache)
  curated_path <- file.path(cache, "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
  custom_path <- file.path(cache, "my-model.gguf")
  writeBin(charToRaw("GGUFaaaa"), curated_path)
  writeBin(charToRaw("GGUFbbbb"), custom_path)

  models <- llamacpp_list_models(cache)
  expect_equal(nrow(models), 2)
  expect_true(any(models$curated))
  expect_true(any(models$preset == "qwen_3b"))
  expect_true(any(models$name == "my-model.gguf"))
})

test_that("model presets catalog exposes aliases", {
  presets <- llamacpp_model_presets()
  expect_true(all(c("id", "label", "aliases", "filename", "url", "family", "size_gb", "description") %in% names(presets)))
  expect_true("deepseek" %in% unlist(presets$aliases))
  expect_true("starcoder" %in% unlist(presets$aliases))
})

test_that("chat constructor validates model path early", {
  expect_error(chat_llamacpp(model = tempfile(fileext = ".gguf")), "does not exist")
})
