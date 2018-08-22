cmake_minimum_required (VERSION 3.8)

# TODO: CHANGE PROJECT NAME HERE
set(MY_PROJECT_NAME cuDigraph)



set(CUDA_64_BIT_DEVICE_CODE ON)

project(${MY_PROJECT_NAME})

FILE(GLOB MY_INC_DIR "inc" "inc/other_authors")
include_directories("${MY_INC_DIR}")


FILE(GLOB MY_SRC src/*.cxx src/*.cu)
FILE(GLOB MY_HEADERS inc/*.h)
FILE(GLOB MY_PCH_OBJ src/pch.cxx)

message(" ... SRC= ${MY_SRC}")
message(" ... HEADERS= ${MY_HEADERS}")
message(" ... PCH OBJ= ${MY_PCH_OBJ}")

enable_language(CXX)

# ADD CUDA/CXX support

find_package(CUDA)

# enable_language(CUDA) does not respect ${CUDA_TOOLKIT_ROOT_DIR} !!

# add target exe

CUDA_ADD_EXECUTABLE(${MY_PROJECT_NAME} ${MY_SRC} ${MY_HEADERS})

# include pre-compiled headers

include(PrecompiledHeader.cmake)
add_precompiled_header(${MY_PROJECT_NAME} "pch.h"  SOURCE_CXX "${MY_PCH_OBJ}" FORCEINCLUDE )



# setup C++ 11 standard

target_compile_features(${MY_PROJECT_NAME} PUBLIC cxx_std_11)
set_property(TARGET ${MY_PROJECT_NAME} PROPERTY CUDA_STANDARD 11)


# BOOST libraries

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)

add_definitions( -DBOOST_ALL_NO_LIB )

# TODO: setup components

find_package(Boost 1.67.0 COMPONENTS program_options regex thread signals system filesystem chrono date_time atomic ) 

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    target_link_libraries(${MY_PROJECT_NAME} ${Boost_LIBRARIES})
endif()

