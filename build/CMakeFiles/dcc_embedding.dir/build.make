# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.28.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.28.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/jeff/Desktop/Project2_YZ

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/jeff/Desktop/Project2_YZ/build

# Include any dependencies generated for this target.
include CMakeFiles/dcc_embedding.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/dcc_embedding.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/dcc_embedding.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dcc_embedding.dir/flags.make

CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.o: CMakeFiles/dcc_embedding.dir/flags.make
CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.o: /Users/jeff/Desktop/Project2_YZ/src/dnn_embedding.cpp
CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.o: CMakeFiles/dcc_embedding.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/jeff/Desktop/Project2_YZ/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.o -MF CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.o.d -o CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.o -c /Users/jeff/Desktop/Project2_YZ/src/dnn_embedding.cpp

CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jeff/Desktop/Project2_YZ/src/dnn_embedding.cpp > CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.i

CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jeff/Desktop/Project2_YZ/src/dnn_embedding.cpp -o CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.s

CMakeFiles/dcc_embedding.dir/src/matchings.cpp.o: CMakeFiles/dcc_embedding.dir/flags.make
CMakeFiles/dcc_embedding.dir/src/matchings.cpp.o: /Users/jeff/Desktop/Project2_YZ/src/matchings.cpp
CMakeFiles/dcc_embedding.dir/src/matchings.cpp.o: CMakeFiles/dcc_embedding.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/jeff/Desktop/Project2_YZ/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/dcc_embedding.dir/src/matchings.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dcc_embedding.dir/src/matchings.cpp.o -MF CMakeFiles/dcc_embedding.dir/src/matchings.cpp.o.d -o CMakeFiles/dcc_embedding.dir/src/matchings.cpp.o -c /Users/jeff/Desktop/Project2_YZ/src/matchings.cpp

CMakeFiles/dcc_embedding.dir/src/matchings.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/dcc_embedding.dir/src/matchings.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jeff/Desktop/Project2_YZ/src/matchings.cpp > CMakeFiles/dcc_embedding.dir/src/matchings.cpp.i

CMakeFiles/dcc_embedding.dir/src/matchings.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/dcc_embedding.dir/src/matchings.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jeff/Desktop/Project2_YZ/src/matchings.cpp -o CMakeFiles/dcc_embedding.dir/src/matchings.cpp.s

CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.o: CMakeFiles/dcc_embedding.dir/flags.make
CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.o: /Users/jeff/Desktop/Project2_YZ/src/csv_util.cpp
CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.o: CMakeFiles/dcc_embedding.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/jeff/Desktop/Project2_YZ/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.o -MF CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.o.d -o CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.o -c /Users/jeff/Desktop/Project2_YZ/src/csv_util.cpp

CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/jeff/Desktop/Project2_YZ/src/csv_util.cpp > CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.i

CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/jeff/Desktop/Project2_YZ/src/csv_util.cpp -o CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.s

# Object files for target dcc_embedding
dcc_embedding_OBJECTS = \
"CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.o" \
"CMakeFiles/dcc_embedding.dir/src/matchings.cpp.o" \
"CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.o"

# External object files for target dcc_embedding
dcc_embedding_EXTERNAL_OBJECTS =

/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: CMakeFiles/dcc_embedding.dir/src/dnn_embedding.cpp.o
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: CMakeFiles/dcc_embedding.dir/src/matchings.cpp.o
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: CMakeFiles/dcc_embedding.dir/src/csv_util.cpp.o
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: CMakeFiles/dcc_embedding.dir/build.make
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_gapi.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_stitching.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_alphamat.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_aruco.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_bgsegm.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_bioinspired.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_ccalib.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_dnn_objdetect.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_dnn_superres.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_dpm.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_face.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_freetype.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_fuzzy.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_hfs.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_img_hash.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_intensity_transform.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_line_descriptor.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_mcc.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_quality.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_rapid.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_reg.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_rgbd.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_saliency.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_sfm.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_stereo.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_structured_light.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_superres.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_surface_matching.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_tracking.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_videostab.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_viz.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_wechat_qrcode.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_xfeatures2d.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_xobjdetect.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_xphoto.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_shape.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_highgui.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_datasets.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_plot.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_text.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_ml.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_phase_unwrapping.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_optflow.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_ximgproc.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_video.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_videoio.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_imgcodecs.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_objdetect.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_calib3d.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_dnn.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_features2d.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_flann.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_photo.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_imgproc.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: /usr/local/lib/libopencv_core.4.9.0.dylib
/Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding: CMakeFiles/dcc_embedding.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/jeff/Desktop/Project2_YZ/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable /Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dcc_embedding.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dcc_embedding.dir/build: /Users/jeff/Desktop/Project2_YZ/bin/dcc_embedding
.PHONY : CMakeFiles/dcc_embedding.dir/build

CMakeFiles/dcc_embedding.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dcc_embedding.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dcc_embedding.dir/clean

CMakeFiles/dcc_embedding.dir/depend:
	cd /Users/jeff/Desktop/Project2_YZ/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/jeff/Desktop/Project2_YZ /Users/jeff/Desktop/Project2_YZ /Users/jeff/Desktop/Project2_YZ/build /Users/jeff/Desktop/Project2_YZ/build /Users/jeff/Desktop/Project2_YZ/build/CMakeFiles/dcc_embedding.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/dcc_embedding.dir/depend

