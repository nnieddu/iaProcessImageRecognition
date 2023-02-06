cmake_minimum_required( VERSION 3.6 )

# Create Project
project( iaObjectDectorInProcess VERSION 0.1.0)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 17)
set(APP_ICON_RESOURCE_WINDOWS "./ico/appicon.rc")

# set(CMAKE_BUILD_TYPE Debug) #<------------------------------
# set(CMAKE_BUILD_TYPE Release) #<------------------------------

include(CTest)
include(CPack)
enable_testing()
add_executable(iaObjectDectorInProcess main.cpp screenshot.cpp detector.cpp ${APP_ICON_RESOURCE_WINDOWS})

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Additional Include Directories
include_directories( ${OpenCV_INCLUDE_DIRS} )
# Additional Library Directories
link_directories( ${OpenCV_LIB_DIR} )
# Additional Dependencies
target_link_libraries( iaObjectDectorInProcess ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT}) # ${CMAKE_THREAD_LIBS_INIT} )



# message(STATUS "EXPORT ALL NECESSARY DLL TO PUBLISH APP")

# add_custom_command(TARGET iaObjectDectorInProcess POST_BUILD
# 	COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 			"${CUDA_TOOLKIT_ROOT_DIR}/bin/CUBLAS64_12.DLL"
# 			"${CMAKE_BINARY_DIR}/release"
# 	COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 			"${CUDA_TOOLKIT_ROOT_DIR}/bin/CUBLASLT64_12.DLL"
# 			"${CMAKE_BINARY_DIR}/release"
# 	COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 			"${TORCH_LIB_DIR}/CUDNN64_8.DLL"
# 			"${CMAKE_BINARY_DIR}/release"
# 	COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 			"${OPENCV_BIN_DIR}/OPENCV_CORE470.DLL"
# 			"${CMAKE_BINARY_DIR}/release"
# 	COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 			"${OPENCV_BIN_DIR}/OPENCV_DNN470.DLL"
# 			"${CMAKE_BINARY_DIR}/release"
# 	COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 			"${OPENCV_BIN_DIR}/OPENCV_HIGHGUI470.DLL"
# 			"${CMAKE_BINARY_DIR}/release"
# 	COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 			"${OPENCV_BIN_DIR}/OPENCV_IMGCODECS470.DLL"
# 			"${CMAKE_BINARY_DIR}/release"
# 	COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 			"${OPENCV_BIN_DIR}/OPENCV_IMGPROC470.DLL"
# 			"${CMAKE_BINARY_DIR}/release"
# 	COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 			"${OPENCV_BIN_DIR}/OPENCV_VIDEOIO470.DLL"
# 			"${CMAKE_BINARY_DIR}/release"
# 	COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 	"${PYTHON_BIN_DIR}/VCRUNTIME140_1.DLL"
# 	"${CMAKE_BINARY_DIR}/release")



message( STATUS -------------------------------------------- )
message( STATUS -------------------------------------------- )
message( STATUS "    OpenCV_DIR: ${OpenCV_DIR}" )
message( STATUS "                                          " )
message( STATUS "    version: ${OpenCV_VERSION}" )
message( STATUS -------------------------------------------- )
message( STATUS "                                          " )
message( STATUS "    libraries: ${OpenCV_LIBS}" )
message( STATUS -------------------------------------------- )
message( STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" )
message( STATUS -------------------------------------------- )
message( STATUS ------------ DLL --------------------------- )
message( STATUS -------------------------------------------- )
message( STATUS "    CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}" )
message( STATUS "    TORCH_LIB_DIR: ${TORCH_LIB_DIR}" )
message( STATUS "    OPENCV_BIN_DIR: ${OPENCV_BIN_DIR}" )
message( STATUS "    PYTHON_BIN_DIR: ${PYTHON_BIN_DIR}" )
message( STATUS -------------------------------------------- )
message( STATUS "                                          " )