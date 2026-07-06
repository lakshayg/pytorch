# Opt-in build-time profiling via the CMake instrumentation API
# (https://cmake.org/cmake/help/latest/manual/cmake-instrumentation.7.html).
#
# When USE_CMAKE_INSTRUMENTATION is ON, every configure/generate step,
# compile, link, custom command, and install step is timed by CMake. After
# each configure and each build, tools/build_instrumentation.py collates the
# data under <build>/build_profile/runs/<index>/:
#   - summary.txt   per-target compile/link/install/custom timing table
#                   (also printed to the terminal)
#   - trace.json    Chrome trace; open in https://ui.perfetto.dev
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
# need no guards. Requires CMake >= 4.0 (experimental before 4.3) and the
# Ninja or Makefiles generators.

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

if(CMAKE_VERSION VERSION_LESS 4.0)
  message(WARNING "USE_CMAKE_INSTRUMENTATION requires CMake >= 4.0 "
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

# Instrumentation is experimental before CMake 4.3: it must be unlocked with
# a version-specific UUID, and the install/test hooks have older names.
set(_instrumentation_install_hook postCMakeInstall)
if(CMAKE_VERSION VERSION_LESS 4.2)
  set(CMAKE_EXPERIMENTAL_INSTRUMENTATION "a37d1069-1972-4901-b9c9-f194aaf2b6e0")
  set(_instrumentation_install_hook postInstall)
elseif(CMAKE_VERSION VERSION_LESS 4.3)
  set(CMAKE_EXPERIMENTAL_INSTRUMENTATION "ec7aa2dc-b87f-45a3-8022-fe01c5f59984")
endif()

# postGenerate reports configure/generate timing right after configure.
# postBuild covers direct ninja/make runs, including the pip flow which
# installs via `cmake --build . --target install`; postCMakeBuild and the
# install hook cover `cmake --build` / `cmake --install` wrappers. Indexes
# with no new data are ignored by the callback.
cmake_instrumentation(
  API_VERSION 1
  DATA_VERSION 1
  HOOKS postGenerate postBuild postCMakeBuild ${_instrumentation_install_hook}
  CALLBACK "${Python_EXECUTABLE}" "${_TORCH_INSTRUMENTATION_SCRIPT}" collect
           --out-dir "${_TORCH_INSTRUMENTATION_OUT_DIR}"
)
