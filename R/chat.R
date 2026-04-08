LlamaCppChat <- R6::R6Class(
  "Chat",
  public = list(
    initialize = function(provider, system_prompt = NULL, echo = c("none", "output", "all")) {
      private$provider <- provider
      private$echo <- .llamacpp_match_echo(echo)
      private$turns <- list()
      private$tools <- list()
      private$callback_on_tool_request <- function(request) invisible(request)
      private$callback_on_tool_result <- function(result) invisible(result)
      class(self) <- c("llama_cpp_chat", "Chat", "R6")
      if (!is.null(system_prompt)) {
        private$turns <- list(ellmer::Turn("system", contents = list(ellmer::ContentText(system_prompt))))
      }
    },

    get_turns = function(include_system_prompt = FALSE) {
      turns <- private$turns
      if (isTRUE(include_system_prompt)) {
        return(turns)
      }
      Filter(function(turn) !identical(turn@role, "system"), turns)
    },

    set_turns = function(value) {
      stopifnot(is.list(value))
      private$turns <- value
      invisible(self)
    },

    add_turn = function(user = NULL, system = NULL) {
      if (!is.null(system)) {
        private$turns <- c(private$turns, list(ellmer::Turn("system", contents = list(ellmer::ContentText(system)))))
      }
      if (!is.null(user)) {
        private$turns <- c(private$turns, list(.ellmer_ns_get("user_turn")(user)))
      }
      invisible(self)
    },

    get_system_prompt = function() {
      system_turn <- private$find_last_turn("system")
      if (is.null(system_turn)) NULL else system_turn@text
    },

    set_system_prompt = function(value) {
      private$turns <- Filter(function(turn) !identical(turn@role, "system"), private$turns)
      if (!is.null(value)) {
        private$turns <- c(list(ellmer::Turn("system", contents = list(ellmer::ContentText(value)))), private$turns)
      }
      invisible(self)
    },

    get_model = function() {
      private$provider$model
    },

    get_provider = function() {
      private$provider
    },

    close = function() {
      if (isTRUE(private$closed)) {
        return(invisible(FALSE))
      }

      cpp_llamacpp_session_destroy(private$provider$ptr)
      private$closed <- TRUE
      private$provider$ptr <- NULL
      invisible(TRUE)
    },

    register_tool = function(tool) {
      private$tools[[private$tool_name(tool)]] <- tool
      invisible(self)
    },

    register_tools = function(tools) {
      for (tool in tools) {
        self$register_tool(tool)
      }
      invisible(self)
    },

    get_tools = function() {
      private$tools
    },

    set_tools = function(tools) {
      private$tools <- list()
      self$register_tools(tools)
    },

    on_tool_request = function(callback) {
      private$callback_on_tool_request <- callback
      invisible(self)
    },

    on_tool_result = function(callback) {
      private$callback_on_tool_result <- callback
      invisible(self)
    },

    last_turn = function(role = c("assistant", "user", "system")) {
      private$find_last_turn(match.arg(role))
    },

    chat = function(..., echo = NULL) {
      private$ensure_open()
      echo <- .llamacpp_match_echo(echo %||% private$echo)
      user_turn <- .ellmer_ns_get("user_turn")(...)
      private$turns <- c(private$turns, list(user_turn))
      result <- private$run_chat(stream = FALSE)
      if (echo != "none") {
        cat(result$text)
        if (!endsWith(result$text, "\n")) {
          cat("\n")
        }
        return(invisible(result$text))
      }
      result$text
    },

    stream = function(..., stream = c("text", "content")) {
      private$ensure_open()
      stream <- match.arg(stream)
      user_turn <- .ellmer_ns_get("user_turn")(...)
      private$turns <- c(private$turns, list(user_turn))
      private$run_chat(stream = TRUE, yield_as_content = identical(stream, "content"))
    },

    chat_async = function(...) {
      cli::cli_abort("`chat_async()` is not supported yet for llamacppR.")
    },

    chat_structured = function(...) {
      cli::cli_abort("`chat_structured()` is not supported yet for llamacppR.")
    },

    chat_structured_async = function(...) {
      cli::cli_abort("`chat_structured_async()` is not supported yet for llamacppR.")
    },

    stream_async = function(...) {
      cli::cli_abort("`stream_async()` is not supported yet for llamacppR.")
    },

    extract_data = function(...) {
      cli::cli_abort("`extract_data()` is not supported yet for llamacppR.")
    },

    extract_data_async = function(...) {
      cli::cli_abort("`extract_data_async()` is not supported yet for llamacppR.")
    },

    get_tokens = function(include_system_prompt = FALSE) {
      private$ensure_open()
      tokens <- private$token_log
      if (is.null(tokens)) {
        return(data.frame(input = numeric(), output = numeric(), stringsAsFactors = FALSE))
      }
      if (isTRUE(include_system_prompt)) {
        tokens
      } else {
        tokens
      }
    },

    get_cost = function(include = c("all", "last")) {
      private$ensure_open()
      include <- match.arg(include)
      NA_real_
    }
  ),
  private = list(
    provider = NULL,
    turns = NULL,
    tools = NULL,
    echo = "none",
    closed = FALSE,
    callback_on_tool_request = NULL,
    callback_on_tool_result = NULL,
    token_log = NULL,

    ensure_open = function() {
      if (isTRUE(private$closed) || is.null(private$provider$ptr)) {
        cli::cli_abort("This llama.cpp chat has been closed. Create a new chat to continue.")
      }
    },

    find_last_turn = function(role) {
      turns <- Filter(function(turn) identical(turn@role, role), rev(private$turns))
      if (length(turns) == 0) NULL else turns[[1]]
    },

    tool_name = function(tool) {
      if (!is.null(attr(tool, "name", exact = TRUE))) {
        return(attr(tool, "name", exact = TRUE))
      }
      if (inherits(tool, "S7_object") && !is.null(tool@name)) {
        return(tool@name)
      }
      nm <- deparse(substitute(tool))
      if (identical(nm, "tool")) {
        cli::cli_abort("Registered tools need an explicit name.")
      }
      nm
    },

    tool_description = function(tool) {
      if (inherits(tool, "S7_object") && !is.null(tool@description)) {
        tool@description
      } else {
        ""
      }
    },

    tool_args = function(tool) {
      fmls <- names(formals(tool))
      fmls <- setdiff(fmls, "...")
      if (length(fmls) == 0) "" else paste(fmls, collapse = ", ")
    },

    tool_instructions = function() {
      if (length(private$tools) == 0) {
        return("")
      }

      lines <- vapply(private$tools, function(tool) {
        sprintf(
          "- %s: %s Args: %s",
          private$tool_name(tool),
          private$tool_description(tool),
          private$tool_args(tool)
        )
      }, character(1))

      paste(
        "You may call tools when they are needed.",
        "Available tools:",
        paste(lines, collapse = "\n"),
        paste(
          "If you need a tool, respond with JSON only in this exact shape:",
          '{"tool_calls":[{"name":"tool_name","arguments":{"arg":"value"}}]}'
        ),
        "If no tool is needed, answer normally in plain text.",
        sep = "\n\n"
      )
    },

    turn_to_role_content = function(turn) {
      role <- turn@role
      contents <- turn@contents

      if (length(contents) == 0) {
        return(list(role = role, content = ""))
      }

      if (all(vapply(contents, .ellmer_ns_get("is_tool_result"), logical(1)))) {
        role <- "tool"
      }

      content <- vapply(contents, function(content) {
        if (inherits(content, "ellmer::ContentText")) {
          content@text
        } else if (.ellmer_ns_get("is_tool_request")(content)) {
          jsonlite::toJSON(
            list(tool_call = list(id = content@id, name = content@name, arguments = content@arguments)),
            auto_unbox = TRUE
          )
        } else if (.ellmer_ns_get("is_tool_result")(content)) {
          paste0(
            "Tool ",
            content@request@name,
            " result:\n",
            .ellmer_ns_get("tool_string")(content)
          )
        } else {
          as.character(content)
        }
      }, character(1))

      list(role = role, content = paste(content, collapse = "\n\n"))
    },

    prompt_messages = function() {
      turns <- private$turns
      if (length(private$tools) > 0) {
        extra <- private$tool_instructions()
        system_turn <- private$find_last_turn("system")
        if (is.null(system_turn)) {
          turns <- c(list(ellmer::Turn("system", contents = list(ellmer::ContentText(extra)))), turns)
        } else {
          turns[[which(vapply(turns, function(x) identical(x@role, "system"), logical(1)))[1]]] <-
            ellmer::Turn("system", contents = list(ellmer::ContentText(paste(system_turn@text, extra, sep = "\n\n"))))
        }
      }

      lapply(turns, private$turn_to_role_content)
    },

    invoke_tools = function(tool_requests) {
      results <- lapply(tool_requests, function(request) {
        private$callback_on_tool_request(request)
        result <- .ellmer_ns_get("invoke_tool")(request)
        private$callback_on_tool_result(result)
        result
      })

      tool_turn <- .ellmer_ns_get("tool_results_as_turn")(results)
      if (!is.null(tool_turn)) {
        private$turns <- c(private$turns, list(tool_turn))
      }
      results
    },

    parse_tool_calls = function(text) {
      if (length(private$tools) == 0) {
        return(NULL)
      }

      candidates <- c(
        text,
        gsub("^```json\\s*|\\s*```$", "", text),
        sub(".*?(\\{[[:space:][:print:]]*\\})[[:space:]]*$", "\\1", text)
      )

      for (candidate in unique(candidates)) {
        parsed <- tryCatch(jsonlite::fromJSON(candidate, simplifyVector = FALSE), error = function(e) NULL)
        if (is.null(parsed) || is.null(parsed$tool_calls)) {
          next
        }

        requests <- lapply(parsed$tool_calls, function(call) {
          tool <- private$tools[[call$name]]
          ellmer::ContentToolRequest(
            id = call$id %||% paste0("tool_", call$name),
            name = call$name,
            arguments = call$arguments %||% list(),
            tool = tool
          )
        })
        return(requests)
      }

      NULL
    },

    generate_once = function(stream = FALSE) {
      private$ensure_open()
      messages <- private$prompt_messages()
      roles <- vapply(messages, `[[`, character(1), "role")
      contents <- vapply(messages, `[[`, character(1), "content")
      prompt <- cpp_llamacpp_apply_chat_template(private$provider$ptr, roles, contents, TRUE)

      cpp_llamacpp_session_generate(
        private$provider$ptr,
        prompt,
        as.integer(private$provider$params$max_tokens %||% 512L),
        as.numeric(private$provider$params$temperature %||% 0.8),
        as.integer(private$provider$params$top_k %||% 40L),
        as.numeric(private$provider$params$top_p %||% 0.95),
        as.numeric(private$provider$params$min_p %||% 0.05),
        as.numeric(private$provider$params$repeat_penalty %||% 1.1),
        as.integer(private$provider$seed %||% -1L)
      )
    },

    append_assistant_text_turn = function(text, tokens) {
      turn <- .ellmer_ns_get("assistant_turn")(contents = list(ellmer::ContentText(text)))
      private$turns <- c(private$turns, list(turn))
      private$token_log <- rbind(
        private$token_log,
        data.frame(
          provider = "llama.cpp",
          model = basename(private$provider$model),
          input = tokens$prompt_tokens %||% NA_real_,
          output = tokens$completion_tokens %||% NA_real_,
          stringsAsFactors = FALSE
        )
      )
      turn
    },

    append_assistant_tool_turn = function(requests, tokens) {
      turn <- .ellmer_ns_get("assistant_turn")(contents = requests)
      private$turns <- c(private$turns, list(turn))
      private$token_log <- rbind(
        private$token_log,
        data.frame(
          provider = "llama.cpp",
          model = basename(private$provider$model),
          input = tokens$prompt_tokens %||% NA_real_,
          output = tokens$completion_tokens %||% NA_real_,
          stringsAsFactors = FALSE
        )
      )
      turn
    },

    run_chat = function(stream = FALSE, yield_as_content = FALSE) {
      if (!stream) {
        for (i in seq_len(5L)) {
          result <- private$generate_once(stream = FALSE)
          requests <- private$parse_tool_calls(result$text)
          if (length(requests) == 0) {
            private$append_assistant_text_turn(result$text, result)
            return(result)
          }

          private$append_assistant_tool_turn(requests, result)
          private$invoke_tools(requests)
        }

        cli::cli_abort("Tool-calling exceeded the maximum number of rounds.")
      }

      coro::generator(function() {
        for (i in seq_len(5L)) {
          result <- private$generate_once(stream = TRUE)
          requests <- private$parse_tool_calls(result$text)
          if (length(requests) == 0) {
            private$append_assistant_text_turn(result$text, result)
            for (piece in result$pieces) {
              if (yield_as_content) {
                yield(ellmer::ContentText(piece))
              } else {
                yield(piece)
              }
            }
            return(invisible())
          }

          private$append_assistant_tool_turn(requests, result)
          private$invoke_tools(requests)
        }

        cli::cli_abort("Tool-calling exceeded the maximum number of rounds.")
      })
    }
  )
)

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) y else x
}

chat_llamacpp <- function(system_prompt = NULL,
                          model,
                          seed = NULL,
                          params = ellmer::params(),
                          echo = c("none", "output", "all"),
                          n_ctx = 2048L,
                          n_batch = n_ctx,
                          n_threads = 0L,
                          n_gpu_layers = 0L) {
  .llamacpp_validate_model_path(model)
  echo <- .llamacpp_match_echo(echo)

  ptr <- cpp_llamacpp_session_create(
    normalizePath(model, winslash = "/", mustWork = TRUE),
    as.integer(n_ctx),
    as.integer(n_batch),
    as.integer(n_threads),
    as.integer(n_gpu_layers)
  )

  provider <- list(
    ptr = ptr,
    model = normalizePath(model, winslash = "/", mustWork = TRUE),
    seed = seed,
    params = params,
    n_ctx = n_ctx,
    n_batch = n_batch,
    n_threads = n_threads,
    n_gpu_layers = n_gpu_layers
  )

  LlamaCppChat$new(provider = provider, system_prompt = system_prompt, echo = echo)
}
