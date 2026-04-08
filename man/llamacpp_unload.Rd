\name{llamacpp_unload}
\alias{llamacpp_unload}
\title{Unload a llama.cpp session}
\usage{
llamacpp_unload(x)
}
\arguments{
  \item{x}{A chat object created by \code{chat_llamacpp()} or a native session pointer.}
}
\description{
Explicitly releases the native llama.cpp model and context associated with a
chat or session object.
}
