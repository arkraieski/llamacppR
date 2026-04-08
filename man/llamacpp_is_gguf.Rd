\name{llamacpp_is_gguf}
\alias{llamacpp_is_gguf}
\title{Check whether a file is GGUF}
\usage{
llamacpp_is_gguf(path)
}
\arguments{
  \item{path}{Path to inspect.}
}
\description{
Validates the magic bytes at the start of a file to determine whether it looks
like a GGUF model.
}
