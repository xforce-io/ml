cmake_minimum_required(VERSION 2.6.4)

project(plsa)

include_directories(
  /usr/include/
  ${CMAKE_SOURCE_DIR}/deps/public-cpp/include/
  /usr/local/include
  /home/freeman/software/include/
)

link_directories(
  ${CMAKE_SOURCE_DIR}/deps/public-cpp/lib/public-cpp/
  /usr/local/lib/
  /home/freeman/software/lib/
)

FILE(GLOB_RECURSE PLSA_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -W -Wall -Wno-deprecated -Wno-invalid-offsetof -Wno-strict-aliasing -O3 -std=c++11")
set(LIB_DEPS ${CMAKE_SOURCE_DIR}/deps/public-cpp/lib/public-cpp/libpublic-cpp.a)

add_executable(plsa
  ${PLSA_FILES})

target_link_libraries(plsa
  ${LIB_DEPS}
  pthread
  rt
  log4cplus
  z
  public-cpp
)
