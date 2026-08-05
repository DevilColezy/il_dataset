cmake_minimum_required(VERSION 3.0.2)
project(il_dataset)

# ── C++17 requirement ───────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep Eigen's architecture-dependent ABI consistent with flightlib.  The
# pybind module allocates flightlib::Quadrotor and also exchanges Eigen-bearing
# local-planner types with il_local_planner_lib, so every il_dataset C++ target
# must use the same architecture flags as flightlib.
if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
  set(IL_DATASET_ARCH_FLAGS
    -Wno-psabi
    -march=armv7-a
    -mfpu=neon
    -mfloat-abi=hard
    -funsafe-math-optimizations
  )
elseif("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
  set(IL_DATASET_ARCH_FLAGS
    -Wno-psabi
    -march=armv8-a+crypto
    -mcpu=cortex-a57+crypto
  )
else()
  set(IL_DATASET_ARCH_FLAGS -march=native)
endif()

# ── Catkin ──────────────────────────────────────────────────────
find_package(catkin REQUIRED COMPONENTS
  flightlib
  rospy
  std_msgs
)

# ── Eigen3 ──────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED)

# ── OpenCV (required by flightlib headers) ──────────────────────
find_package(OpenCV REQUIRED)

# Optional external nonlinear optimizer.  The planner remains buildable on
# systems without NLopt and falls back to its bounded native optimizer.
find_package(NLopt QUIET)

# ── pybind11 ────────────────────────────────────────────────────
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _pybind11_ret
  )
  if(_pybind11_ret EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}/pybind11Config.cmake")
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 REQUIRED)
  else()
    execute_process(
      COMMAND python3 -c "import os, pybind11; print(os.path.join(os.path.dirname(pybind11.__file__), 'share', 'cmake', 'pybind11'))"
      OUTPUT_VARIABLE PYBIND11_CMAKE_DIR2
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(EXISTS "${PYBIND11_CMAKE_DIR2}/pybind11Config.cmake")
      list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR2}")
      find_package(pybind11 REQUIRED)
    else()
      message(FATAL_ERROR "pybind11 not found. Install: pip3 install pybind11  OR  sudo apt install pybind11-dev")
    endif()
  endif()
endif()

# ── Python3 development ─────────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ── Include directories ─────────────────────────────────────────
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

# ── C++ local planner static library ────────────────────────────
set(LOCAL_PLANNER_SOURCES
  src/local_planner/esdf_grid.cpp
  src/local_planner/local_planner.cpp
  src/local_planner/spline_planner.cpp
)

add_library(il_local_planner_lib STATIC ${LOCAL_PLANNER_SOURCES})
set_target_properties(il_local_planner_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(il_local_planner_lib PRIVATE
  -O3 -Wall -Wextra ${IL_DATASET_ARCH_FLAGS}
)
target_link_libraries(il_local_planner_lib ${catkin_LIBRARIES})
if(NLopt_FOUND)
  target_compile_definitions(il_local_planner_lib PRIVATE IL_DATASET_HAS_NLOPT=1)

  # Ubuntu/ROS distributions ship different FindNLopt implementations.
  # Some expose an imported target without propagating its include directory;
  # others only expose one of these legacy variables.  Add every populated
  # spelling explicitly (empty entries are harmless).
  target_include_directories(il_local_planner_lib PRIVATE
    ${NLOPT_INCLUDE_DIR}
    ${NLOPT_INCLUDE_DIRS}
    ${NLopt_INCLUDE_DIRS}
  )

  if(TARGET NLopt::nlopt)
    target_link_libraries(il_local_planner_lib NLopt::nlopt)
  elseif(TARGET nlopt)
    target_link_libraries(il_local_planner_lib nlopt)
  else()
    target_link_libraries(il_local_planner_lib
      ${NLOPT_LIBRARIES}
      ${NLopt_LIBRARIES}
    )
  endif()
  message(STATUS "il_dataset: NLopt enabled for local trajectory optimization")
  message(STATUS "il_dataset: NLopt include dirs: ${NLOPT_INCLUDE_DIR};${NLOPT_INCLUDE_DIRS};${NLopt_INCLUDE_DIRS}")
else()
  message(STATUS "il_dataset: NLopt not found; using native bounded optimizer")
endif()

# ── pybind11 Python module ──────────────────────────────────────
pybind11_add_module(_il_local_planner
  src/local_planner/pybind_module.cpp
  src/local_planner/depth_integrator.cpp
  src/local_planner/observed_map_ops.cpp
)
target_compile_options(_il_local_planner PRIVATE
  -O3 -Wall -Wextra ${IL_DATASET_ARCH_FLAGS}
)
target_link_libraries(_il_local_planner PRIVATE
  il_local_planner_lib
  ${catkin_LIBRARIES}
  stdc++fs
)

if(CATKIN_ENABLE_TESTING)
  add_executable(
    il_local_planner_trajectory_test
    tests/test_trajectory_consistency.cpp
  )
  target_compile_options(
    il_local_planner_trajectory_test PRIVATE
    -O2 -Wall -Wextra ${IL_DATASET_ARCH_FLAGS}
  )
  target_link_libraries(
    il_local_planner_trajectory_test
    il_local_planner_lib
    ${catkin_LIBRARIES}
  )
  add_test(
    NAME il_local_planner_trajectory_test
    COMMAND il_local_planner_trajectory_test
  )
endif()

# Place the .so next to the Python scripts so "import _il_local_planner" works
set_target_properties(_il_local_planner PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts"
)

# ── Catkin package (minimal — no install, works in devel space) ─
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES il_local_planner_lib
)

catkin_install_python(PROGRAMS
  scripts/expert_behavior_catalog.py
  scripts/expert_behavior_suite.py
  scripts/expert_behavior_test.py
  scripts/expert_behavior_visualizer.py
  scripts/observation_rollout_scenario_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

