\name{llamacpp_download_default_model}
\alias{llamacpp_download_default_model}
\title{Download a curated default GGUF model}
\usage{
llamacpp_download_default_model(
  model = c("3b", "0.5b", "starcoder", "deepseek"),
  path = NULL,
  force = FALSE
)
}
\arguments{
  \item{model}{Which curated default model to download.}
  \item{path}{Destination path for the downloaded model.}
  \item{force}{Whether to overwrite an existing file.}
}
\description{
Downloads a curated GGUF model from Hugging Face and returns the local path.
}
