cmake_minimum_required(VERSION 3.0.2)
project(il_dataset)

# ── C++17 requirement ───────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep Eigen's architecture-dependent ABI consistent with flightlib.  The
# _flightmare_dynamics module allocates flightlib::Quadrotor and the
# hierarchical-expert library exchanges Eigen types with flightlib, so every
# il_dataset C++ target must use the same architecture flags as flightlib.
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
  # R29r: MUST stay ABI-consistent with the prebuilt flightlib (compiled
  # with the DEFAULT architecture).  -march=native enabled AVX (32-byte)
  # Eigen alignment in this module while flightlib::Quadrotor uses 16-byte
  # alignment, so the cross-.so Quadrotor new/delete pair mismatched and
  # ~Quadrotor's Eigen::aligned_free double-freed (crash on the first
  # FlightmareDynamicsBridge dealloc: "double free or corruption (out)").
  # Use the default architecture exactly like flightlib.
  set(IL_DATASET_ARCH_FLAGS "")
endif()

# ── AddressSanitizer (optional native crash diagnosis) ────────────────
# Enable this option to locate the exact out-of-bounds WRITE in the C++
# hierarchical-expert / FlightmareDynamics code:
#   catkin build il_dataset --cmake-args -DIL_DATASET_ASAN=ON
# then re-run; AddressSanitizer prints the offending file:line.  Turn it
# OFF again for normal collection (ASan is slower).
option(IL_DATASET_ASAN "Build C++ local planner with AddressSanitizer" OFF)
set(IL_DATASET_SAN_FLAGS "")
if(IL_DATASET_ASAN)
  set(IL_DATASET_SAN_FLAGS -fsanitize=address -fno-omit-frame-pointer)
  set(IL_DATASET_OPT_FLAGS -O1 -g)
  message(STATUS "il_dataset: AddressSanitizer ENABLED (native heap diagnosis)")
else()
  set(IL_DATASET_OPT_FLAGS -O3)
endif()

# ── Catkin ──────────────────────────────────────────────────────
find_package(catkin REQUIRED COMPONENTS
  flightlib
  rospy
  std_msgs
)

# ── Eigen3 ──────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED)

# ── Threads (parallel task preflight: std::async / std::future) ──
find_package(Threads REQUIRED)

# ── OpenCV (required by flightlib headers) ──────────────────────
find_package(OpenCV REQUIRED)

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

# ── Hierarchical local expert (v9 port from il_2d_multiscale_debug) ─
# Pure C++17 + Eigen (NO Qt, NO ROS headers, NO truth input at runtime).
# Owns: 5 Hz VisibilityTargetCorrector + 30 Hz LocalPlanner30Hz +
# EffectiveTargetAdapter + HierarchicalExpertFsm + ObservedGrid2D +
# Flightmare2DObservation (depth->grid) + PreflightSimulator.
set(HIERARCHICAL_EXPERT_SOURCES
  src/hierarchical_expert/observed_grid_2d.cpp
  src/hierarchical_expert/effective_target_adapter.cpp
  src/hierarchical_expert/macro_expert_5hz.cpp
  src/hierarchical_expert/local_planner_30hz.cpp
  src/hierarchical_expert/ego_bspline.cpp
  src/hierarchical_expert/hierarchical_expert_fsm.cpp
  src/hierarchical_expert/flightmare_2d_observation.cpp
  src/hierarchical_expert/hierarchical_expert.cpp
  src/hierarchical_expert/preflight_simulator.cpp
  # ── offline Scene/Task Blueprint pipeline (il_dataset-feature-v4) ─
  src/hierarchical_expert/scene_task_blueprint.cpp
  src/hierarchical_expert/scene_profile_generator.cpp
  src/hierarchical_expert/scene_geometry_cache.cpp
  src/hierarchical_expert/task_candidate_generator.cpp
  src/hierarchical_expert/depth_proxy_evaluator.cpp
  src/hierarchical_expert/distribution_analyzer.cpp
  src/hierarchical_expert/route_qualifier.cpp
  src/hierarchical_expert/blueprint_generation_controller.cpp
)

add_library(il_hierarchical_expert_lib STATIC ${HIERARCHICAL_EXPERT_SOURCES})
set_target_properties(il_hierarchical_expert_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(il_hierarchical_expert_lib PRIVATE
  ${IL_DATASET_OPT_FLAGS} -Wall -Wextra ${IL_DATASET_SAN_FLAGS} ${IL_DATASET_ARCH_FLAGS}
)
target_link_libraries(il_hierarchical_expert_lib
  ${catkin_LIBRARIES}
  ${IL_DATASET_SAN_FLAGS}
  Threads::Threads
)

# ── pybind11 Python module for the hierarchical expert ──────────
pybind11_add_module(_il_hierarchical_expert
  src/hierarchical_expert/pybind_module.cpp
)
target_compile_options(_il_hierarchical_expert PRIVATE
  ${IL_DATASET_OPT_FLAGS} -Wall -Wextra ${IL_DATASET_SAN_FLAGS} ${IL_DATASET_ARCH_FLAGS}
  # R29r: hide pybind11 internals so two pybind modules (this one and
  # _flightmare_dynamics) never clash on pybind11_object_dealloc / type_info
  # across .so boundaries (caused double-free in FlightmareDynamicsBridge
  # dealloc).  Only the PYBIND11_MODULE init symbol stays exported.
  -fvisibility=hidden
)
target_link_libraries(_il_hierarchical_expert PRIVATE
  il_hierarchical_expert_lib
  ${catkin_LIBRARIES}
  ${IL_DATASET_SAN_FLAGS}
  Threads::Threads
  stdc++fs
)

# Place the .so next to the Python scripts so "import _il_hierarchical_expert"
# works (same convention as the other modules).
set_target_properties(_il_hierarchical_expert PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts"
)

# ── Standalone FlightmareDynamics bridge (production) ─────────────
# The ONLY flightlib consumer from Python.  Independent of the (legacy)
# old expert stack so the manager never loads old expert modules just to
# drive the real quadrotor dynamics.
pybind11_add_module(_flightmare_dynamics
  src/flightmare_dynamics/pybind_module.cpp
)
target_compile_options(_flightmare_dynamics PRIVATE
  ${IL_DATASET_OPT_FLAGS} -Wall -Wextra ${IL_DATASET_SAN_FLAGS} ${IL_DATASET_ARCH_FLAGS}
  # R29r: hide pybind11 internals (see _il_hierarchical_expert note).
  -fvisibility=hidden
)
target_link_libraries(_flightmare_dynamics PRIVATE
  ${catkin_LIBRARIES}
  ${IL_DATASET_SAN_FLAGS}
  stdc++fs
)
set_target_properties(_flightmare_dynamics PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts"
)

# ── Catkin package (minimal — no install, works in devel space) ─
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES il_hierarchical_expert_lib
)

# ── ROS Python node installation ─────────────────────────────────
# Install the manager so `roslaunch il_dataset ...` resolves the node
# executable directly.  Only true ROS executables are listed; the il_*
# helper modules are plain import modules and stay in the source tree.
catkin_install_python(
  PROGRAMS
    scripts/il_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ── Offline blueprint tests / benchmark (optional, opt-in) ────────
# Pure C++17 static sanity tests + raycast micro-benchmark for the
# deficit-driven Scene/Task Blueprint pipeline.  Build with:
#   catkin build il_dataset --cmake-args -DIL_DATASET_BUILD_TESTS=ON
# then run the produced binaries (they are plain executables, no gtest):
#   ./devel/lib/il_dataset/test_blueprint_geometry
#   ./devel/lib/il_dataset/bench_blueprint_raycast
option(IL_DATASET_BUILD_TESTS "Build offline blueprint static tests + benchmark" OFF)
if(IL_DATASET_BUILD_TESTS)
  add_executable(test_blueprint_geometry
    test/test_blueprint_geometry.cpp
  )
  target_compile_options(test_blueprint_geometry PRIVATE
    ${IL_DATASET_OPT_FLAGS} -Wall -Wextra ${IL_DATASET_SAN_FLAGS} ${IL_DATASET_ARCH_FLAGS}
  )
  target_link_libraries(test_blueprint_geometry
    il_hierarchical_expert_lib
    ${catkin_LIBRARIES}
    ${IL_DATASET_SAN_FLAGS}
  )

  add_executable(bench_blueprint_raycast
    bench/bench_blueprint_raycast.cpp
  )
  target_compile_options(bench_blueprint_raycast PRIVATE
    ${IL_DATASET_OPT_FLAGS} -Wall -Wextra ${IL_DATASET_SAN_FLAGS} ${IL_DATASET_ARCH_FLAGS}
  )
  target_link_libraries(bench_blueprint_raycast
    il_hierarchical_expert_lib
    ${catkin_LIBRARIES}
    ${IL_DATASET_SAN_FLAGS}
  )
endif()

