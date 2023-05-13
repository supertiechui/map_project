#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "PATCHWORK::patchworkpp" for configuration "Debug"
set_property(TARGET PATCHWORK::patchworkpp APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(PATCHWORK::patchworkpp PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "Eigen3::Eigen"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libpatchworkpp.so"
  IMPORTED_SONAME_DEBUG "libpatchworkpp.so"
  )

list(APPEND _cmake_import_check_targets PATCHWORK::patchworkpp )
list(APPEND _cmake_import_check_files_for_PATCHWORK::patchworkpp "${_IMPORT_PREFIX}/lib/libpatchworkpp.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
