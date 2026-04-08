\name{llamacpp_model_info}
\alias{llamacpp_model_info}
\title{Inspect a GGUF model through llama.cpp}
\usage{
llamacpp_model_info(
  model,
  n_ctx = 2048L,
  n_batch = n_ctx,
  n_threads = 0L,
  n_gpu_layers = 0L
)
}
\arguments{
  \item{model}{Path to a GGUF file.}
  \item{n_ctx}{Context size used when opening the model.}
  \item{n_batch}{Batch size used when opening the model.}
  \item{n_threads}{Number of CPU threads.}
  \item{n_gpu_layers}{Number of GPU layers to offload when supported.}
}
\description{
Loads a GGUF model through native llama.cpp bindings and returns basic metadata.
}
