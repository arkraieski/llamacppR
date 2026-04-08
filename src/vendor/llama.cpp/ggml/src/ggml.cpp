#include "ggml-impl.h"
#include "../../include/llamacppR_vendor_shim.h"

#include <cstdlib>
#include <exception>

static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    llamacppR_vendor_fatal("uncaught exception in ggml");
    GGML_UNREACHABLE();
}

static bool ggml_uncaught_exception_init = []{
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    return true;
}();
