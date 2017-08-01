#ifndef MRWSI_PLATFORM
#define MRWSI_PLATFORM

#include "bin.h"
#include "problem.h"
#include "resource.h"

typedef struct task_t { item_t item; } task_t;

typedef struct machine_t {
    bin_t bin;
    item_t item;
} machine_t;

typedef struct platform { item_t item; } platform;

#endif /* ifndef MRWSI_PLATFORM */
