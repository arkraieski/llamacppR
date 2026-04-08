args <- commandArgs(trailingOnly = TRUE)

target <- if (length(args) > 0) {
  normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
} else {
  normalizePath("src/llamacppR.so", winslash = "/", mustWork = TRUE)
}

bad_symbols <- c(
  "__Exit",
  "_Exit",
  "___stderrp",
  "___stdoutp",
  "_stderr",
  "_stdout",
  "_printf",
  "_puts",
  "_putchar",
  "_abort"
)

nm_output <- system2("nm", c("-u", target), stdout = TRUE, stderr = TRUE)

symbols <- trimws(sub("^.*\\s", "", nm_output))
matches <- unique(nm_output[symbols %in% bad_symbols])

if (length(matches) > 0) {
  writeLines(matches)
  stop("Found disallowed linked symbols in ", target, call. = FALSE)
}

writeLines(sprintf("No disallowed linked symbols found in %s", target))
