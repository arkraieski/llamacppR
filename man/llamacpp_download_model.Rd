\name{llamacpp_download_model}
\alias{llamacpp_download_model}
\title{Download a curated model preset}
\usage{
llamacpp_download_model(
  model = c("qwen_3b", "qwen_0_5b", "starcoder", "deepseek"),
  path = NULL,
  force = FALSE
)
}
\arguments{
  \item{model}{Preset id or alias.}
  \item{path}{Destination path for the downloaded model.}
  \item{force}{Whether to overwrite an existing file.}
}
\description{
Downloads one of the curated model presets shipped with llamacppR.
}
