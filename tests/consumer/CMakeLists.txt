# Downstream-consumer check: builds against an *installed* loki, exactly as a
# user project would. Run by CI after `cmake --install`:
#   cmake -S tests/consumer -B build-consumer -DCMAKE_PREFIX_PATH=<prefix>
#   cmake --build build-consumer && ./build-consumer/loki_consumer
cmake_minimum_required(VERSION 3.28...4.4)
project(loki_consumer LANGUAGES CXX)

find_package(loki CONFIG REQUIRED)

add_executable(loki_consumer main.cpp)
target_link_libraries(loki_consumer PRIVATE loki::loki)
