add_executable(dsgb10_rank main.cpp cmd.cpp serve.cpp metrics.cpp)
target_link_libraries(dsgb10_rank PRIVATE dsmodel)

find_package(Threads REQUIRED)
target_link_libraries(dsgb10_rank PRIVATE Threads::Threads)

# Does not link the model or CUDA; also buildable manually on a CPU-only host.
add_executable(pico_metrics_test EXCLUDE_FROM_ALL metrics.cpp tests/metrics_test.cpp)
target_link_libraries(pico_metrics_test PRIVATE Threads::Threads)
