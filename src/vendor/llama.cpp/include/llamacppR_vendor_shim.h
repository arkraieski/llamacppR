#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*llamacppR_vendor_log_callback_t)(int level, const char * text, void * user_data);
typedef void (*llamacppR_vendor_fatal_callback_t)(const char * text, void * user_data);

void llamacppR_vendor_set_log_callback(
    llamacppR_vendor_log_callback_t callback,
    void * user_data
);

void llamacppR_vendor_set_fatal_callback(
    llamacppR_vendor_fatal_callback_t callback,
    void * user_data
);

void llamacppR_vendor_log(int level, const char * text);
void llamacppR_vendor_fatal(const char * text);
const char * llamacppR_vendor_last_fatal_message(void);
void llamacppR_vendor_clear_last_fatal_message(void);

#ifdef __cplusplus
}
#endif
