cmake_minimum_required(VERSION 3.17)
project(function1)

set(CMAKE_CXX_STANDARD 17)

#include_directories("inc" "inc/CBS" "inc/PIBT" "CBS" "PIBT")
#file(GLOB SOURCES "CBS/*.cpp" "PIBT/*.cpp" "inc/*.h" "inc/PIBT/*.h" "inc/CBS/*.h" "*.cpp" )
include_directories("inc")
file(GLOB SOURCES "*.cpp" )

add_executable(function1 ${SOURCES})

# Find Boost
find_package(Boost REQUIRED COMPONENTS program_options system filesystem)

# Find Eigen3 for PIBT
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


include_directories( ${Boost_INCLUDE_DIRS} )
target_link_libraries(function1 ${Boost_LIBRARIES} Eigen3::Eigen)
