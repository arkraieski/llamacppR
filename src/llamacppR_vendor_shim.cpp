#include <R_ext/Error.h>

#include <mutex>
#include <string>

#include "llamacppR_vendor_shim.h"

namespace {

std::mutex shim_mutex;
llamacppR_vendor_log_callback_t log_callback = nullptr;
void * log_callback_user_data = nullptr;
llamacppR_vendor_fatal_callback_t fatal_callback = nullptr;
void * fatal_callback_user_data = nullptr;
std::string last_fatal_message;

} // namespace

extern "C" void llamacppR_vendor_set_log_callback(
    llamacppR_vendor_log_callback_t callback,
    void * user_data
) {
  std::lock_guard<std::mutex> lock(shim_mutex);
  log_callback = callback;
  log_callback_user_data = user_data;
}

extern "C" void llamacppR_vendor_set_fatal_callback(
    llamacppR_vendor_fatal_callback_t callback,
    void * user_data
) {
  std::lock_guard<std::mutex> lock(shim_mutex);
  fatal_callback = callback;
  fatal_callback_user_data = user_data;
}

extern "C" void llamacppR_vendor_log(int level, const char * text) {
  llamacppR_vendor_log_callback_t callback = nullptr;
  void * user_data = nullptr;

  {
    std::lock_guard<std::mutex> lock(shim_mutex);
    callback = log_callback;
    user_data = log_callback_user_data;
  }

  if (callback != nullptr && text != nullptr) {
    callback(level, text, user_data);
  }
}

extern "C" void llamacppR_vendor_fatal(const char * text) {
  llamacppR_vendor_fatal_callback_t callback = nullptr;
  void * user_data = nullptr;

  {
    std::lock_guard<std::mutex> lock(shim_mutex);
    last_fatal_message = text == nullptr ? "llama.cpp fatal error" : text;
    callback = fatal_callback;
    user_data = fatal_callback_user_data;
  }

  if (callback != nullptr) {
    callback(last_fatal_message.c_str(), user_data);
  }

  Rf_error("%s", last_fatal_message.c_str());
}

extern "C" const char * llamacppR_vendor_last_fatal_message(void) {
  std::lock_guard<std::mutex> lock(shim_mutex);
  return last_fatal_message.c_str();
}

extern "C" void llamacppR_vendor_clear_last_fatal_message(void) {
  std::lock_guard<std::mutex> lock(shim_mutex);
  last_fatal_message.clear();
}
