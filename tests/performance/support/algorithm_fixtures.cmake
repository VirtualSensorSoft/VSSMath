cmake_minimum_required(VERSION 3.28)

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/../algorithm_decimal.cmake")

function(vss_algorithm_fixture_record out_json profile configuration median iterations allocations)
  set(case_ids
    tri.solve.n16.rhs4 tri.solve.n64.rhs4 tri.solve.n192.rhs4
    lu.factor.n16 lu.factor.n64 lu.factor.n128
    lu.solve.n16.rhs4 lu.solve.n64.rhs4 lu.solve.n128.rhs4
    qr_ls.factor.m32.n16 qr_ls.factor.m128.n64 qr_ls.factor.m256.n128
    qr_ls.solve.m32.n16.rhs4 qr_ls.solve.m128.n64.rhs4 qr_ls.solve.m256.n128.rhs4
    qr_square.factor.n192 qr_square.solve.n192.rhs4
    cholesky.factor.n16 cholesky.factor.n64 cholesky.factor.n128
    cholesky.solve.n16.rhs4 cholesky.solve.n64.rhs4 cholesky.solve.n128.rhs4
    eigen.decompose.n16 eigen.decompose.n64 eigen.decompose.n128
    covariance.propagate.j64x128.c128 differentiation.jacobian.in64.out128
    statistics.mean.n4096 statistics.population_variance.n4096
    statistics.sample_variance.n4096 statistics.population_stddev.n4096
    statistics.sample_stddev.n4096 statistics.rms.n4096
    probability.survival.n4096 probability.log_survival.n4096)
  vss_algorithm_decimal_divide_small("${median}" "${iterations}" display remainder)

  set(json "{")
  string(APPEND json "\"schema_version\":2,")
  string(APPEND json "\"profile_id\":\"${profile}\",")
  string(APPEND json "\"environment\":{")
  string(APPEND json
    "\"math_commit\":\"0000000000000000000000000000000000000000\",")
  string(APPEND json "\"package_version\":\"0.1.0\",")
  string(APPEND json "\"benchmark_sha256\":\""
    "00000000" "00000000" "00000000" "00000000"
    "00000000" "00000000" "00000000" "00000000" "\",")
  string(APPEND json "\"configuration\":\"${configuration}\",")
  string(APPEND json "\"compiler_id\":\"GNU\",")
  string(APPEND json "\"compiler_version\":\"13.3.0\",")
  string(APPEND json "\"target_triple\":\"x86_64-linux-gnu\",")
  string(APPEND json "\"stdlib_id\":\"libstdc++\",")
  string(APPEND json "\"stdlib_version\":\"13.3.0\",")
  string(APPEND json "\"cmake_version\":\"3.28.3\",")
  string(APPEND json "\"cxx_standard\":\"20\",")
  string(APPEND json "\"compile_flags\":\"-O3\",")
  string(APPEND json "\"link_flags\":\"\",")
  string(APPEND json "\"sanitizers\":\"off\",")
  string(APPEND json "\"coverage\":\"off\",")
  string(APPEND json "\"os_name\":\"Ubuntu\",")
  string(APPEND json "\"os_version\":\"24.04\",")
  string(APPEND json "\"kernel\":\"6.8\",")
  string(APPEND json "\"architecture\":\"x86_64\",")
  string(APPEND json "\"cpu_vendor\":\"GenuineIntel\",")
  string(APPEND json "\"cpu_model\":\"fixture-cpu\",")
  string(APPEND json "\"cpu_reported_frequency\":\"unavailable\",")
  string(APPEND json "\"frequency_governor\":\"performance\",")
  string(APPEND json "\"turbo_state\":\"enabled\",")
  string(APPEND json "\"runner_image\":\"ubuntu-24.04\",")
  string(APPEND json "\"runner_label_or_host\":\"fixture-runner\",")
  string(APPEND json "\"utc_timestamp\":\"2026-01-01T00:00:00Z\",")
  string(APPEND json "\"logical_cores\":4,")
  string(APPEND json "\"physical_memory_bytes\":1000000},")
  string(APPEND json "\"method\":{")
  string(APPEND json "\"seed\":\"0x5653534d415448\",")
  string(APPEND json "\"warmup_min_operations\":10,")
  string(APPEND json "\"warmup_min_duration_ns\":100000000,")
  string(APPEND json "\"target_batch_duration_ns\":20000000,")
  string(APPEND json "\"samples\":30,\"batch_limit\":1048576,")
  string(APPEND json "\"clock\":\"steady_clock\",")
  string(APPEND json "\"statistic\":\"integer_median\",")
  string(APPEND json "\"dispersion\":\"integer_mad\",")
  string(APPEND json "\"allocation_scope\":\"executable_local\"},")
  string(APPEND json "\"cases\":[")
  set(first TRUE)
  foreach(case_id IN LISTS case_ids)
    if(NOT first)
      string(APPEND json ",")
    endif()
    set(first FALSE)
    string(APPEND json "{\"case_id\":\"${case_id}\",")
    string(APPEND json "\"iterations_per_sample\":${iterations},")
    string(APPEND json "\"batch_durations_ns\":[")
    foreach(sample_index RANGE 1 30)
      if(sample_index GREATER 1)
        string(APPEND json ",")
      endif()
      string(APPEND json "${median}")
    endforeach()
    string(APPEND json "],\"median_batch_ns\":${median},")
    string(APPEND json "\"mad_batch_ns\":0,")
    string(APPEND json "\"median_ns_per_operation\":${display},")
    string(APPEND json "\"allocations_per_operation\":${allocations},")
    string(APPEND json "\"checksum_hex\":\"0x1p+0\",")
    string(APPEND json "\"correctness_passed\":true}")
  endforeach()
  string(APPEND json "]}")
  set("${out_json}" "${json}" PARENT_SCOPE)
endfunction()
