cmake_minimum_required (VERSION 3.0.0)
project (KratosStructuredSolverSandbox)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/libs") # Equivalent to the LD_LIBRARY_PATH (where my library will be created)

SET( CMAKE_CONFIGURATION_TYPES Release FullDebug)
SET( BASIC_DEBUG_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
SET( CMAKE_CXX_FLAGS_FULLDEBUG "${BASIC_DEBUG_FLAGS} -D_DEBUG" )

SET( CMAKE_C_FLAGS_FULLDEBUG "${CMAKE_C_FLAGS_DEBUG}" )
SET( CMAKE_SHARED_LINKER_FLAGS_FULLDEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" )
SET( CMAKE_EXE_LINKER_FLAGS_FULLDEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" )
SET( CMAKE_MODULE_LINKER_FLAGS_FULLDEBUG "${CMAKE_MODULE_LINKER_FLAGS}")

if(NOT CMAKE_BUILD_TYPE)
    if(DEFINED ENV{BUILD_TYPE})
        list(FIND CMAKE_CONFIGURATION_TYPES $ENV{BUILD_TYPE} FOUND_MODE)
        if(FOUND_MODE EQUAL -1)
            message("Unknown CMAKE_BUILD_TYPE '${CMAKE_BUILD_TYPE}', using 'Release'.")
            set(CMAKE_BUILD_TYPE Release)
        else(FOUND_MODE EQUAL -1)
            set(CMAKE_BUILD_TYPE $ENV{BUILD_TYPE})
        endif(FOUND_MODE EQUAL -1)
    else(DEFINED ENV{BUILD_TYPE})
        message("--No CMAKE_BUILD_TYPE is defined, building in 'Release' mode.")
        set(CMAKE_BUILD_TYPE Release)
    endif(DEFINED ENV{BUILD_TYPE})
endif(NOT CMAKE_BUILD_TYPE)

include_directories("includes") # Header files directories
include_directories("external_libraries/eigen-3.4.0") # Include Eigen directories
include_directories("external_libraries/mdspan-mdspan-0.6.0") # Include mdspan directories

file (GLOB_RECURSE KRATOS_STRUCTURED_SOLVER_SANDBOX_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/cell_utilities.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/incompressible_navier_stokes_q1_p0_structured_element.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/mesh_utilities.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/operators.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/runge_kutta_utilities.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/sbm_utilities.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sources/time_utilities.cpp
) # Source files directories

add_library(KratosStructuredSolverSandboxLibrary SHARED ${KRATOS_STRUCTURED_SOLVER_SANDBOX_SOURCES}) # This creates the library (we need to specify all the sources)

add_executable(KratosStructuredSolverSandboxExecutable "${CMAKE_CURRENT_SOURCE_DIR}/sources/run.cpp") # This creates the executable

target_link_libraries(KratosStructuredSolverSandboxExecutable PUBLIC KratosStructuredSolverSandboxLibrary) # This links the static/shared library above with the previous executable

install(TARGETS KratosStructuredSolverSandboxLibrary DESTINATION libs) # This installs the target (library) in the libs folder

install(TARGETS KratosStructuredSolverSandboxExecutable DESTINATION bin) # This installs the target (executable) in the bin folder
