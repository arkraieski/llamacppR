#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"
#include "llamacppR_vendor_shim.h"

using namespace Rcpp;

namespace {

struct session_state {
  llama_model * model = nullptr;
  llama_context * ctx = nullptr;
  std::string model_path;
  int n_ctx = 2048;
  int n_batch = 2048;
  int n_threads = 0;
  int n_gpu_layers = 0;
};

bool backend_initialized = false;

void quiet_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
  (void) user_data;
  if (level >= GGML_LOG_LEVEL_ERROR && text != nullptr) {
    Rcpp::Rcerr << text;
  }
}

void shim_log_callback(int level, const char * text, void * user_data) {
  quiet_log_callback(static_cast<enum ggml_log_level>(level), text, user_data);
}

void shim_fatal_callback(const char * text, void * user_data) {
  (void) user_data;
  if (text == nullptr || std::strlen(text) == 0) {
    Rf_error("llama.cpp fatal error");
  }
  Rf_error("%s", text);
}

void ensure_backend_initialized() {
  if (!backend_initialized) {
    llamacppR_vendor_set_log_callback(shim_log_callback, nullptr);
    llamacppR_vendor_set_fatal_callback(shim_fatal_callback, nullptr);
    llamacppR_vendor_clear_last_fatal_message();
    llama_log_set(quiet_log_callback, nullptr);
    ggml_backend_load_all();
    llama_backend_init();
    backend_initialized = true;
  }
}

session_state * get_session(SEXP ext) {
  Rcpp::XPtr<session_state> ptr(ext);
  if (ptr.get() == nullptr) {
    Rcpp::stop("llama.cpp session has been freed.");
  }
  return ptr.get();
}

void release_session(session_state * session) {
  if (session == nullptr) {
    return;
  }
  if (session->ctx != nullptr) {
    llama_free(session->ctx);
    session->ctx = nullptr;
  }
  if (session->model != nullptr) {
    llama_model_free(session->model);
    session->model = nullptr;
  }
}

std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
  std::vector<char> buf(256);
  int n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
  if (n < 0) {
    buf.resize(static_cast<size_t>(-n));
    n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
  }
  if (n < 0) {
    throw std::runtime_error("failed to decode token piece");
  }
  return std::string(buf.data(), static_cast<size_t>(n));
}

std::vector<llama_token> tokenize_prompt(llama_context * ctx, const std::string & prompt) {
  const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
  int n_tokens = -llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
  if (n_tokens <= 0) {
    throw std::runtime_error("prompt tokenization failed");
  }
  std::vector<llama_token> tokens(static_cast<size_t>(n_tokens));
  int rc = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(), n_tokens, true, true);
  if (rc < 0) {
    throw std::runtime_error("prompt tokenization failed");
  }
  return tokens;
}

llama_sampler * create_sampler(int seed,
                               double temperature,
                               int top_k,
                               double top_p,
                               double min_p,
                               double repeat_penalty) {
  llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());

  if (temperature <= 0) {
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    return chain;
  }

  if (repeat_penalty > 0 && std::abs(repeat_penalty - 1.0) > 1e-8) {
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(64, static_cast<float>(repeat_penalty), 0.0f, 0.0f));
  }
  if (top_k > 0) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
  }
  if (top_p > 0 && top_p < 1.0) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(static_cast<float>(top_p), 1));
  }
  if (min_p > 0) {
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(static_cast<float>(min_p), 1));
  }
  llama_sampler_chain_add(chain, llama_sampler_init_temp(static_cast<float>(temperature)));
  llama_sampler_chain_add(chain, llama_sampler_init_dist(seed < 0 ? LLAMA_DEFAULT_SEED : static_cast<uint32_t>(seed)));

  return chain;
}

} // namespace

// [[Rcpp::export]]
SEXP cpp_llamacpp_session_create(std::string model_path,
                                 int n_ctx,
                                 int n_batch,
                                 int n_threads,
                                 int n_gpu_layers) {
  ensure_backend_initialized();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = n_gpu_layers;

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = static_cast<uint32_t>(n_ctx);
  ctx_params.n_batch = static_cast<uint32_t>(n_batch);
  if (n_threads > 0) {
    ctx_params.n_threads = static_cast<uint32_t>(n_threads);
    ctx_params.n_threads_batch = static_cast<uint32_t>(n_threads);
  }

  session_state * session = new session_state();
  session->model_path = model_path;
  session->n_ctx = n_ctx;
  session->n_batch = n_batch;
  session->n_threads = n_threads;
  session->n_gpu_layers = n_gpu_layers;

  session->model = llama_model_load_from_file(model_path.c_str(), model_params);
  if (session->model == nullptr) {
    delete session;
    Rcpp::stop("Failed to load model from '%s'.", model_path);
  }

  session->ctx = llama_init_from_model(session->model, ctx_params);
  if (session->ctx == nullptr) {
    llama_model_free(session->model);
    delete session;
    Rcpp::stop("Failed to initialize llama.cpp context.");
  }

  Rcpp::XPtr<session_state> ptr(session, true);
  return ptr;
}

// [[Rcpp::export]]
Rcpp::List cpp_llamacpp_session_info(SEXP ext) {
  session_state * session = get_session(ext);

  char desc_buf[512];
  llama_model_desc(session->model, desc_buf, sizeof(desc_buf));

  return Rcpp::List::create(
    Rcpp::Named("model") = session->model_path,
    Rcpp::Named("description") = std::string(desc_buf),
    Rcpp::Named("n_ctx") = session->n_ctx,
    Rcpp::Named("n_batch") = session->n_batch,
    Rcpp::Named("model_size") = static_cast<double>(llama_model_size(session->model)),
    Rcpp::Named("n_ctx_train") = llama_model_n_ctx_train(session->model)
  );
}

// [[Rcpp::export]]
bool cpp_llamacpp_session_destroy(SEXP ext) {
  if (TYPEOF(ext) != EXTPTRSXP) {
    Rcpp::stop("Expected an external pointer.");
  }

  session_state * session = static_cast<session_state *>(R_ExternalPtrAddr(ext));
  if (session == nullptr) {
    return false;
  }

  release_session(session);
  delete session;
  R_ClearExternalPtr(ext);
  return true;
}

// [[Rcpp::export]]
std::string cpp_llamacpp_apply_chat_template(SEXP ext,
                                             Rcpp::CharacterVector roles,
                                             Rcpp::CharacterVector contents,
                                             bool add_generation_prompt) {
  session_state * session = get_session(ext);

  if (roles.size() != contents.size()) {
    Rcpp::stop("roles and contents must have the same length.");
  }

  const int n = roles.size();
  std::vector<std::string> role_storage(static_cast<size_t>(n));
  std::vector<std::string> content_storage(static_cast<size_t>(n));
  std::vector<llama_chat_message> messages(static_cast<size_t>(n));

  for (int i = 0; i < n; ++i) {
    role_storage[static_cast<size_t>(i)] = Rcpp::as<std::string>(roles[i]);
    content_storage[static_cast<size_t>(i)] = Rcpp::as<std::string>(contents[i]);
    messages[static_cast<size_t>(i)] = {
      role_storage[static_cast<size_t>(i)].c_str(),
      content_storage[static_cast<size_t>(i)].c_str()
    };
  }

  const char * tmpl = llama_model_chat_template(session->model, nullptr);
  if (tmpl == nullptr) {
    std::string fallback;
    for (int i = 0; i < n; ++i) {
      fallback += role_storage[static_cast<size_t>(i)] + ": " + content_storage[static_cast<size_t>(i)] + "\n";
    }
    if (add_generation_prompt) {
      fallback += "assistant: ";
    }
    return fallback;
  }

  int needed = llama_chat_apply_template(
    tmpl,
    messages.data(),
    static_cast<int>(messages.size()),
    add_generation_prompt,
    nullptr,
    0
  );
  if (needed < 0) {
    Rcpp::stop("Failed to apply llama.cpp chat template.");
  }

  std::vector<char> buf(static_cast<size_t>(needed) + 1);
  int written = llama_chat_apply_template(
    tmpl,
    messages.data(),
    static_cast<int>(messages.size()),
    add_generation_prompt,
    buf.data(),
    static_cast<int>(buf.size())
  );
  if (written < 0) {
    Rcpp::stop("Failed to apply llama.cpp chat template.");
  }

  return std::string(buf.data(), static_cast<size_t>(written));
}

// [[Rcpp::export]]
Rcpp::List cpp_llamacpp_session_generate(SEXP ext,
                                         std::string prompt,
                                         int max_tokens,
                                         double temperature,
                                         int top_k,
                                         double top_p,
                                         double min_p,
                                         double repeat_penalty,
                                         int seed) {
  session_state * session = get_session(ext);

  llama_memory_clear(llama_get_memory(session->ctx), true);

  std::vector<llama_token> prompt_tokens = tokenize_prompt(session->ctx, prompt);
  if (static_cast<int>(prompt_tokens.size()) >= session->n_ctx) {
    Rcpp::stop("Prompt is too large for the configured context window.");
  }

  int rc = llama_decode(
    session->ctx,
    llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()))
  );
  if (rc != 0) {
    Rcpp::stop("llama_decode() failed for the prompt.");
  }

  const llama_vocab * vocab = llama_model_get_vocab(session->model);
  std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
    create_sampler(seed, temperature, top_k, top_p, min_p, repeat_penalty),
    &llama_sampler_free
  );

  std::vector<std::string> pieces;
  std::string text;
  bool stopped_by_eog = false;

  for (int i = 0; i < max_tokens; ++i) {
    llama_token token = llama_sampler_sample(sampler.get(), session->ctx, -1);
    if (llama_vocab_is_eog(vocab, token)) {
      stopped_by_eog = true;
      break;
    }

    std::string piece = token_to_piece(vocab, token);
    pieces.push_back(piece);
    text += piece;

    llama_sampler_accept(sampler.get(), token);
    int decode_rc = llama_decode(session->ctx, llama_batch_get_one(&token, 1));
    if (decode_rc != 0) {
      Rcpp::stop("llama_decode() failed while generating tokens.");
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("text") = text,
    Rcpp::Named("pieces") = pieces,
    Rcpp::Named("prompt_tokens") = static_cast<int>(prompt_tokens.size()),
    Rcpp::Named("completion_tokens") = static_cast<int>(pieces.size()),
    Rcpp::Named("finish_reason") = stopped_by_eog ? "stop" : "length"
  );
}

// [[Rcpp::export]]
std::string cpp_llamacpp_test_trigger_fatal() {
  ensure_backend_initialized();
  llamacppR_vendor_fatal("llama.cpp fatal test");
  return "unreachable";
}
