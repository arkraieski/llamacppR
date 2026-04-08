llamacpp_model_presets <- function() {
  data.frame(
    id = c("qwen_3b", "qwen_0_5b", "starcoder", "deepseek"),
    label = c(
      "Qwen 2.5 3B Instruct",
      "Qwen 2.5 0.5B Instruct",
      "StarCoder2 3B Instruct",
      "DeepSeek R1 Distill Qwen 7B"
    ),
    aliases = I(list(
      c("3b", "qwen_3b"),
      c("0.5b", "qwen_0_5b"),
      c("starcoder", "starcoder2_3b_instruct"),
      c("deepseek", "deepseek_qwen_7b")
    )),
      filename = c(
        "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "starcoder2-3b-instruct.Q4_K_M.gguf",
        "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
      ),
      url = c(
        paste0(
          "https://huggingface.co/bartowski/",
          "Qwen2.5-3B-Instruct-GGUF/resolve/main/",
          "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
        ),
        paste0(
          "https://huggingface.co/bartowski/",
          "Qwen2.5-0.5B-Instruct-GGUF/resolve/main/",
          "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
        ),
        paste0(
          "https://huggingface.co/QuantFactory/",
          "starcoder2-3b-instruct-GGUF/resolve/main/",
          "starcoder2-3b-instruct.Q4_K_M.gguf"
        ),
        paste0(
          "https://huggingface.co/bartowski/",
          "DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/",
          "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
        )
      ),
    family = c("qwen", "qwen", "starcoder", "deepseek"),
    size_gb = c(1.93, 0.40, 1.89, 4.68),
    description = c(
      "Balanced default for general chat.",
      "Small fast fallback for quick smoke tests.",
      "Code-focused instruct model.",
      "Stronger general-purpose distilled reasoning/chat model."
    ),
    stringsAsFactors = FALSE
  )
}

.llamacpp_resolve_preset <- function(model) {
  presets <- llamacpp_model_presets()
  hits <- vapply(presets$aliases, function(x) model %in% x, logical(1))
  if (!any(hits)) {
    cli::cli_abort(
      c(
        "Unknown curated model preset {.val {model}}.",
        i = "Available presets: {.val {presets$id}}",
        i = "Available aliases: {.val {unlist(presets$aliases)}}"
      )
    )
  }
  presets[which(hits)[1], , drop = FALSE]
}

llamacpp_cache_dir <- function() {
  path <- tools::R_user_dir("llamacppR", which = "cache")
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}

llamacpp_model_path <- function(model = c("qwen_3b", "qwen_0_5b", "starcoder", "deepseek")) {
  model <- match.arg(model, choices = unique(c(llamacpp_model_presets()$id, unlist(llamacpp_model_presets()$aliases))))
  preset <- .llamacpp_resolve_preset(model)
  file.path(llamacpp_cache_dir(), preset$filename[[1]])
}

llamacpp_default_model_path <- function(model = c("3b", "0.5b", "starcoder", "deepseek")) {
  model <- match.arg(model, choices = c("3b", "0.5b", "starcoder", "deepseek"))
  llamacpp_model_path(model)
}

llamacpp_is_gguf <- function(path) {
  if (!is.character(path) || length(path) != 1 || is.na(path) || !file.exists(path)) {
    return(FALSE)
  }

  con <- file(path, open = "rb")
  on.exit(close(con), add = TRUE)
  identical(rawToChar(readBin(con, "raw", n = 4L)), "GGUF")
}

llamacpp_download_model <- function(model = c("qwen_3b", "qwen_0_5b", "starcoder", "deepseek"), path = NULL, force = FALSE) {
  model <- match.arg(model, choices = unique(c(llamacpp_model_presets()$id, unlist(llamacpp_model_presets()$aliases))))
  preset <- .llamacpp_resolve_preset(model)
  if (is.null(path)) {
    path <- llamacpp_model_path(model = model)
  }
  stopifnot(is.character(path), length(path) == 1L, !is.na(path))

  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  if (file.exists(path) && !isTRUE(force)) {
    if (!llamacpp_is_gguf(path)) {
      cli::cli_abort("{.file {path}} already exists but is not a GGUF file.")
    }
    return(normalizePath(path, winslash = "/", mustWork = TRUE))
  }

  tmp <- paste0(path, ".part")
  if (file.exists(tmp)) {
    unlink(tmp)
  }

  curl::curl_download(
    url = preset$url[[1]],
    destfile = tmp,
    quiet = FALSE
  )

  if (!llamacpp_is_gguf(tmp)) {
    unlink(tmp)
    cli::cli_abort("Downloaded file is not a valid GGUF model.")
  }

  ok <- file.rename(tmp, path)
  if (!ok) {
    file.copy(tmp, path, overwrite = TRUE)
    unlink(tmp)
  }

  normalizePath(path, winslash = "/", mustWork = TRUE)
}

llamacpp_download_default_model <- function(model = c("3b", "0.5b", "starcoder", "deepseek"), path = NULL, force = FALSE) {
  model <- match.arg(model, choices = c("3b", "0.5b", "starcoder", "deepseek"))
  llamacpp_download_model(model = model, path = path, force = force)
}

llamacpp_model_info <- function(model, n_ctx = 2048L, n_batch = n_ctx, n_threads = 0L, n_gpu_layers = 0L) {
  .llamacpp_validate_model_path(model)
  ptr <- cpp_llamacpp_session_create(
    normalizePath(model, winslash = "/", mustWork = TRUE),
    as.integer(n_ctx),
    as.integer(n_batch),
    as.integer(n_threads),
    as.integer(n_gpu_layers)
  )
  cpp_llamacpp_session_info(ptr)
}

llamacpp_list_models <- function(path = llamacpp_cache_dir(), recursive = TRUE) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)

  files <- list.files(
    path = path,
    pattern = "\\.gguf$",
    recursive = recursive,
    full.names = TRUE
  )

  if (length(files) == 0) {
    return(data.frame(
      name = character(),
      path = character(),
      size_bytes = numeric(),
      curated = logical(),
      preset = character(),
      stringsAsFactors = FALSE
    ))
  }

  curated <- llamacpp_model_presets()
  curated_names <- curated$filename
  curated_presets <- curated$id

  info <- file.info(files)
  names_only <- basename(files)
  idx <- match(names_only, curated_names)

  data.frame(
    name = names_only,
    path = normalizePath(files, winslash = "/", mustWork = TRUE),
    size_bytes = unname(info$size),
    curated = !is.na(idx),
    preset = ifelse(is.na(idx), "", curated_presets[idx]),
    stringsAsFactors = FALSE
  )
}

llamacpp_unload <- function(x) {
  if (inherits(x, "LlamaCppChat")) {
    x$close()
    return(invisible(TRUE))
  }

  cpp_llamacpp_session_destroy(x)
  invisible(TRUE)
}

.llamacpp_validate_model_path <- function(model) {
  if (missing(model) || is.null(model) || !nzchar(model)) {
    cli::cli_abort("`model` must be a path to a local GGUF file.")
  }
  if (!file.exists(model)) {
    cli::cli_abort("Model file {.file {model}} does not exist.")
  }
  if (!llamacpp_is_gguf(model)) {
    cli::cli_abort("Model file {.file {model}} is not a GGUF file.")
  }
}

.llamacpp_match_echo <- function(echo = c("none", "output", "all")) {
  match.arg(echo)
}

.ellmer_ns_get <- function(name) {
  get(name, envir = asNamespace("ellmer"), inherits = FALSE)
}
