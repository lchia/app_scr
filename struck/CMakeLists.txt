cmake_minimum_required(VERSION 2.6)
project(struck)

include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11) 
#if (CHECK_CXX_COMPILER_FLAG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#endif()
 
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_HOME_DIRECTORY}/cmake ${CMAKE_MODULE_PATH})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    src
    ../sifs/test
    ../sifs/src
    ../sifs/libs
    ${EIGEN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# add all .cpp and .h files in src
file(GLOB_RECURSE SRC RELATIVE ${PROJECT_SOURCE_DIR} src/*.cpp)
file(GLOB_RECURSE HEADERS RELATIVE ${PROJECT_SOURCE_DIR} src/*.h)
#file(GLOB_RECURSE SRC RELATIVE ${PROJECT_SOURCE_DIR} ../sifs/test/ssvm_sifs.cpp)
#file(GLOB_RECURSE HEADERS RELATIVE ${PROJECT_SOURCE_DIR} ../sifs/src/*.h)
#file(GLOB_RECURSE HEADERS RELATIVE ${PROJECT_SOURCE_DIR} ../sifs/libs/sdm/lib/*.hpp)

add_executable(struck ${HEADERS} ${SRC})

target_link_libraries(struck
    ${OpenCV_LIBS}
)
