\name{llamacpp_model_path}
\alias{llamacpp_model_path}
\title{Get the cache path for a curated model preset}
\usage{
llamacpp_model_path(model = c("qwen_3b", "qwen_0_5b", "starcoder", "deepseek"))
}
\arguments{
  \item{model}{Preset id or alias.}
}
\description{
Returns the local cache path used by llamacppR for a curated model preset.
}
