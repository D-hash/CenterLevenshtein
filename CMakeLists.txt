cmake_minimum_required(VERSION 3.22)
project(levenshteincpp)

set(CMAKE_CXX_STANDARD 20)

include_directories(/opt/ibm/ILOG/CPLEX_Studio2211/cplex/include /opt/ibm/ILOG/CPLEX_Studio2211/concert/include)
add_executable(levenshteincpp levenshtein.cpp)

target_link_libraries(levenshteincpp /opt/ibm/ILOG/CPLEX_Studio2211/cplex/lib/x86-64_linux/ /opt/ibm/ILOG/CPLEX_Studio2211/concert/lib/x86-64_linux/ /opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/ /opt/ibm/ILOG/CPLEX_Studio2211/opl/bin/x86-64_linux /opt/ibm/ILOG/CPLEX_Studio2211/opl/lib/x86-64_linux/static_pic/ /opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux -lconcert -lilocplex -lcplex -lpthread -ldl)
