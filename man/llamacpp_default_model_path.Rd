\name{llamacpp_default_model_path}
\alias{llamacpp_default_model_path}
\title{Get the cache path for a curated default model}
\usage{
llamacpp_default_model_path(model = c("3b", "0.5b", "starcoder", "deepseek"))
}
\arguments{
  \item{model}{Which curated default model path to return.}
}
\description{
Returns the local cache path used by llamacppR for one of the curated default
GGUF model presets.
}
