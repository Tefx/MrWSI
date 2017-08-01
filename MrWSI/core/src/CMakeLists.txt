include_directories(${MrWSI_SOURCE_DIR}/include)

set(SOURCE_FILES_RESOURCE resource.c)
set(SOURCE_FILES_BIN bin.c mempool.c)
set(SOURCE_FILES_PROBLEM problem.c)

add_library(mrwsi SHARED ${SOURCE_FILES_RESOURCE}
                         ${SOURCE_FILES_BIN}
                         ${SOURCE_FILES_PROBLEM})
