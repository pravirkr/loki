add_executable(tests loki_t.cpp)
target_link_libraries(tests PUBLIC loki Catch2::Catch2WithMain)

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(tests)
