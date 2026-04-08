\name{chat_llamacpp}
\alias{chat_llamacpp}
\title{Create an ellmer-style llama.cpp chat}
\usage{
chat_llamacpp(
  system_prompt = NULL,
  model,
  seed = NULL,
  params = ellmer::params(),
  echo = c("none", "output", "all"),
  n_ctx = 2048L,
  n_batch = n_ctx,
  n_threads = 0L,
  n_gpu_layers = 0L
)
}
\arguments{
  \item{system_prompt}{Optional system prompt.}
  \item{model}{Path to a local GGUF model file.}
  \item{seed}{Optional seed forwarded to llama.cpp sampling.}
  \item{params}{An \code{ellmer::params()} list.}
  \item{echo}{Whether to echo generated output.}
  \item{n_ctx}{Context size.}
  \item{n_batch}{Batch size used for prompt evaluation.}
  \item{n_threads}{CPU threads used by llama.cpp.}
  \item{n_gpu_layers}{Number of layers to offload to GPU when supported.}
}
\description{
Creates a local chat object backed by native \code{llama.cpp} inference while
following the \code{ellmer} chat API style.
}
