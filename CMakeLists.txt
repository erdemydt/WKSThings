cmake_minimum_required(VERSION 3.16)
project(gp_playground CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG v2.4.0)
FetchContent_MakeAvailable(polyscope)


add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE polyscope)