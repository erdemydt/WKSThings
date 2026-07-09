cmake_minimum_required(VERSION 3.16)
project(gp_playground CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG v2.4.0)
FetchContent_MakeAvailable(polyscope)


FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0)
set(EIGEN_BUILD_DOC OFF)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
FetchContent_MakeAvailable(eigen)

FetchContent_Declare(libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG v2.5.0)
FetchContent_MakeAvailable(libigl)

add_executable(app 
  src/main.cpp
  src/mesh.cpp
  src/operators.cpp
  )
target_link_libraries(app PRIVATE polyscope Eigen3::Eigen igl::core)