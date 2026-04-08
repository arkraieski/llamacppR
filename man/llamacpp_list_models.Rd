\name{llamacpp_list_models}
\alias{llamacpp_list_models}
\title{List local GGUF models in the llamacppR cache}
\usage{
llamacpp_list_models(path = llamacpp_cache_dir(), recursive = TRUE)
}
\arguments{
  \item{path}{Directory to scan for GGUF files.}
  \item{recursive}{Whether to scan subdirectories recursively.}
}
\description{
Lists GGUF files found in the local llamacppR cache directory and marks
whether they match one of the curated default model presets.
}
