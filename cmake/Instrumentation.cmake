# Opt-in build-time profiling via the CMake instrumentation API
# (https://cmake.org/cmake/help/latest/manual/cmake-instrumentation.7.html).
#
# When USE_CMAKE_INSTRUMENTATION is ON, every configure/generate step,
# compile, link, custom command, and install step is timed by CMake. After
# each configure and each build, tools/build_instrumentation.py collates the
# data under <build>/build_profile/runs/<index>/:
#   - summary.txt   per-target compile/link/install/custom timing table
#                   (also printed to the terminal)
#   - trace.json    CMake-generated Google trace; open in
#                   https://ui.perfetto.dev
#   - raw snippet files as emitted by CMake
#
# Build steps that Ninja folds into another edge (e.g. POST_BUILD commands,
# which are appended to the link command and therefore counted inside the
# "link" snippet) can be timed separately by prefixing them with the launcher
# from torch_instrumentation_launcher():
#
#   torch_instrumentation_launcher(_launcher "my-step-name")
#   add_custom_command(TARGET tgt POST_BUILD
#     COMMAND ${_launcher} <real command...> VERBATIM)
#
# The launcher expands to nothing when instrumentation is off, so call sites
# need no guards. Requires CMake >= 4.3 and the Ninja or Makefiles
# generators.

option(USE_CMAKE_INSTRUMENTATION "Collect a build time profile via the CMake instrumentation API" OFF)

get_filename_component(_TORCH_INSTRUMENTATION_SCRIPT
  "${CMAKE_CURRENT_LIST_DIR}/../tools/build_instrumentation.py" ABSOLUTE)
set(_TORCH_INSTRUMENTATION_OUT_DIR "${CMAKE_BINARY_DIR}/build_profile")

function(torch_instrumentation_launcher out_var step_name)
  if(NOT USE_CMAKE_INSTRUMENTATION)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()
  set(${out_var}
    "${Python_EXECUTABLE}" "${_TORCH_INSTRUMENTATION_SCRIPT}" time-step
    --name "${step_name}" --out-dir "${_TORCH_INSTRUMENTATION_OUT_DIR}" --
    PARENT_SCOPE)
endfunction()

# CMake's instrumentation launcher embeds every declared output of a custom
# command into a single ctest argument. Commands with very large output
# lists (e.g. ATen codegen, hundreds of generated files) exceed the kernel's
# 128KiB per-argument limit and fail with "Argument list too long". Until
# this is fixed upstream, call this to opt the calling directory's custom
# commands (subdirectories included) out of instrumentation; they run
# unwrapped and produce no timing data.
function(torch_instrumentation_disable_custom_commands)
  if(USE_CMAKE_INSTRUMENTATION)
    set_property(DIRECTORY PROPERTY RULE_LAUNCH_CUSTOM "")
  endif()
endfunction()

if(NOT USE_CMAKE_INSTRUMENTATION)
  return()
endif()

if(CMAKE_VERSION VERSION_LESS 4.3)
  message(WARNING "USE_CMAKE_INSTRUMENTATION requires CMake >= 4.3 "
                  "(found ${CMAKE_VERSION}); disabling.")
  set(USE_CMAKE_INSTRUMENTATION OFF)
  return()
endif()

if(NOT CMAKE_GENERATOR MATCHES "Ninja|Makefiles" OR WIN32)
  message(WARNING "USE_CMAKE_INSTRUMENTATION only supports the Ninja and "
                  "Makefiles generators on non-Windows hosts; disabling.")
  set(USE_CMAKE_INSTRUMENTATION OFF)
  return()
endif()

# postGenerate reports configure/generate timing right after configure.
# postBuild covers direct ninja/make runs; postCMakeBuild covers `cmake
# --build`, which is what the pip flow uses (`cmake --build . --target
# install`, so its install scripts land in the same index). postCMakeInstall
# covers standalone `cmake --install`. The pre* hooks sweep snippets left
# over from interrupted builds into their own run so they don't skew the
# next build's profile. Indexes with no new data are ignored by the
# callback.
cmake_instrumentation(
  API_VERSION 1
  DATA_VERSION 1
  HOOKS postGenerate preBuild postBuild preCMakeBuild postCMakeBuild
        postCMakeInstall
  OPTIONS trace staticSystemInformation
  CALLBACK "${Python_EXECUTABLE}" "${_TORCH_INSTRUMENTATION_SCRIPT}" collect
           --out-dir "${_TORCH_INSTRUMENTATION_OUT_DIR}"
           --src-dir "${CMAKE_SOURCE_DIR}"
)
