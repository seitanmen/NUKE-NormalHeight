# NukeConfig.cmake — Locate the Nuke NDK
#
# Usage:
#   set(NUKE_ROOT "C:/Program Files/Nuke15.1v1")  # or set NUKE_ROOT env var
#   find_package(Nuke REQUIRED)
#
# This defines the imported INTERFACE target  Nuke::NDK
# and the helper function                       add_nuke_plugin()

# ---------- resolve NUKE_ROOT ------------------------------------------------

if(NOT NUKE_ROOT)
    if(DEFINED ENV{NUKE_ROOT})
        set(NUKE_ROOT "$ENV{NUKE_ROOT}")
    endif()
endif()

# As a last resort, scan common install locations
if(NOT NUKE_ROOT)
    if(WIN32)
        file(GLOB _nuke_dirs "C:/Program Files/Nuke1[5-9]*")
    elseif(APPLE)
        file(GLOB _nuke_dirs "/Applications/Nuke1[5-9]*")
    else()
        file(GLOB _nuke_dirs "/usr/local/Nuke1[5-9]*")
    endif()
    if(_nuke_dirs)
        list(SORT _nuke_dirs COMPARE NATURAL ORDER DESCENDING)
        list(GET _nuke_dirs 0 NUKE_ROOT)
        message(STATUS "Auto-detected Nuke at: ${NUKE_ROOT}")
    endif()
    unset(_nuke_dirs)
endif()

if(NOT NUKE_ROOT)
    message(FATAL_ERROR
        "Nuke installation not found.\n"
        "Set NUKE_ROOT or add it to CMAKE_PREFIX_PATH.\n"
        "Example: cmake -DNUKE_ROOT=/usr/local/Nuke15.1v1 ..")
endif()

# ---------- resolve include / lib directories --------------------------------

# Headers live at  NUKE_ROOT/include/DDImage/...
if(NOT NUKE_INCLUDE_DIR)
    if(EXISTS "${NUKE_ROOT}/include/DDImage/Iop.h")
        set(NUKE_INCLUDE_DIR "${NUKE_ROOT}/include")
    else()
        message(FATAL_ERROR "Nuke headers not found under ${NUKE_ROOT}/include/")
    endif()
endif()

# Platform-specific library names
if(WIN32)
    set(_ddimage_lib  "${NUKE_ROOT}/DDImage.lib")
    set(_extra_libs   "glew32.lib;opengl32.lib;RIPFramework.lib;tbb.lib;tbbmalloc.lib")
elseif(APPLE)
    set(_ddimage_lib  "${NUKE_ROOT}/libDDImage.dylib")
    set(_extra_libs   "")
else()
    set(_ddimage_lib  "${NUKE_ROOT}/libDDImage.so")
    set(_extra_libs   "")
endif()

if(NOT EXISTS "${_ddimage_lib}")
    message(FATAL_ERROR "DDImage library not found at: ${_ddimage_lib}")
endif()

# ---------- imported target --------------------------------------------------

if(NOT TARGET Nuke::NDK)
    add_library(Nuke::NDK INTERFACE IMPORTED)

    set_target_properties(Nuke::NDK PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NUKE_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES      "${_ddimage_lib};${_extra_libs}"
        INTERFACE_COMPILE_FEATURES    "cxx_std_17"
    )

    # The bare-name libs above (glew32.lib, tbb.lib, ...) live in NUKE_ROOT;
    # add it as a link search dir so the linker can resolve them.
    if(WIN32)
        set_property(TARGET Nuke::NDK APPEND PROPERTY
            INTERFACE_LINK_DIRECTORIES "${NUKE_ROOT}")
    endif()

    # Nuke 14.1+ requires this on Linux
    if(UNIX AND NOT APPLE)
        set_property(TARGET Nuke::NDK APPEND PROPERTY
            INTERFACE_COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=0")
    endif()
endif()

# ---------- helper function --------------------------------------------------

function(add_nuke_plugin PLUGIN_NAME)
    add_library(${PLUGIN_NAME} MODULE ${ARGN})
    target_link_libraries(${PLUGIN_NAME} PRIVATE Nuke::NDK)
    target_compile_features(${PLUGIN_NAME} PRIVATE cxx_std_17)

    # Nuke expects no "lib" prefix and platform-specific extension
    set_target_properties(${PLUGIN_NAME} PROPERTIES
        PREFIX ""
        DEBUG_POSTFIX ""
        RELWITHDEBINFO_POSTFIX ""
    )

    if(WIN32)
        set_target_properties(${PLUGIN_NAME} PROPERTIES SUFFIX ".dll")
        # Nuke 16 NDK headers (fnVC.h) #error out unless these are defined.
        target_compile_definitions(${PLUGIN_NAME} PRIVATE
            NOMINMAX
            _USE_MATH_DEFINES
            WIN32_LEAN_AND_MEAN
            _CRT_SECURE_NO_WARNINGS
        )
    elseif(APPLE)
        set_target_properties(${PLUGIN_NAME} PROPERTIES SUFFIX ".dylib")
    else()
        set_target_properties(${PLUGIN_NAME} PROPERTIES SUFFIX ".so")
    endif()
endfunction()

message(STATUS "Nuke NDK found: ${NUKE_ROOT}")
