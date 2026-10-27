cmake_minimum_required(VERSION 3.28)

# RED-phase deterministic host-adapter tests. Production inspection is not
# performed; every fixture lives below the binary tree.
set(checker "${CMAKE_CURRENT_LIST_DIR}/check_local_capture_environment.cmake")
set(work "${CMAKE_CURRENT_BINARY_DIR}/performance-local-capture-fixtures")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")

# The reviewed source preview deliberately has no .git directory. Exercise the
# clean-commit boundary using a disposable synthetic repository in the build tree.
set(source_dir "${work}/source")
file(MAKE_DIRECTORY "${source_dir}")
execute_process(COMMAND git -C "${source_dir}" init --quiet
  RESULT_VARIABLE git_init_status)
if(NOT git_init_status EQUAL 0)
  message(FATAL_ERROR "could not initialize synthetic source repository")
endif()
execute_process(COMMAND git -C "${source_dir}" config user.name "VSSMath test")
execute_process(COMMAND git -C "${source_dir}" config user.email "vssmath-test@example.invalid")
file(WRITE "${source_dir}/README" "synthetic clean source revision\n")
execute_process(COMMAND git -C "${source_dir}" add README)
execute_process(COMMAND git -C "${source_dir}" commit --quiet -m "fixture source"
  RESULT_VARIABLE git_commit_status)
if(NOT git_commit_status EQUAL 0)
  message(FATAL_ERROR "could not commit synthetic source fixture")
endif()
execute_process(
  COMMAND git -C "${source_dir}" rev-parse HEAD
  OUTPUT_VARIABLE expected_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE git_rev_status)
string(LENGTH "${expected_commit}" expected_commit_length)
if(NOT git_rev_status EQUAL 0 OR NOT expected_commit MATCHES "^[0-9a-f]+$" OR
   NOT expected_commit_length EQUAL 40)
  message(FATAL_ERROR "synthetic source commit identity is invalid")
endif()
set(profile_id performance-primary-gcc-debug)
set(configuration Debug)
set(trace_pass
  "0|0.50|55000|2026-10-01T00:00:00Z;10|0.50|55000|2026-10-01T00:00:10Z;20|0.50|55000|2026-10-01T00:00:20Z;30|0.50|55000|2026-10-01T00:00:30Z;40|0.50|55000|2026-10-01T00:00:40Z;50|0.50|55000|2026-10-01T00:00:50Z;60|0.50|55000|2026-10-01T00:01:00Z")
set(trace_high_load
  "0|0.51|55000|2026-10-01T00:00:00Z;10|0.51|55000|2026-10-01T00:00:10Z;20|0.51|55000|2026-10-01T00:00:20Z;30|0.51|55000|2026-10-01T00:00:30Z;40|0.51|55000|2026-10-01T00:00:40Z;50|0.51|55000|2026-10-01T00:00:50Z;60|0.51|55000|2026-10-01T00:01:00Z")
set(trace_reset
  "0|0.50|55000|2026-10-01T00:00:00Z;10|0.50|55000|2026-10-01T00:00:10Z;20|0.50|55000|2026-10-01T00:00:20Z;30|0.51|55000|2026-10-01T00:00:30Z;40|0.50|55000|2026-10-01T00:00:40Z;50|0.50|55000|2026-10-01T00:00:50Z;60|0.50|55000|2026-10-01T00:01:00Z")
set(trace_hot
  "0|0.50|55001|2026-10-01T00:00:00Z;10|0.50|55001|2026-10-01T00:00:10Z;20|0.50|55001|2026-10-01T00:00:20Z;30|0.50|55001|2026-10-01T00:00:30Z;40|0.50|55001|2026-10-01T00:00:40Z;50|0.50|55001|2026-10-01T00:00:50Z;60|0.50|55001|2026-10-01T00:01:00Z")

set(trace_timeout
  "0|0.51|55000|2026-10-01T00:00:00Z;901|0.51|55000|2026-10-01T00:15:01Z")
function(write_profile path)
  file(WRITE "${path}"
    "set(VSS_PERF_PROFILE_SCHEMA_VERSION 1)\n"
    "set(VSS_PERF_HOST_PROFILE_ID performance-primary-local-linux-v1)\n"
    "set(VSS_PERF_OS_NAME Linux)\nset(VSS_PERF_ARCHITECTURE x86_64)\n"
    "set(VSS_PERF_CPU_MODEL \"Synthetic test processor\")\n"
    "set(VSS_PERF_LOGICAL_CPU_COUNT 8)\nset(VSS_PERF_AFFINITY_CPU 4)\n"
    "set(VSS_PERF_AFFINITY_SIBLINGS 0,4)\nset(VSS_PERF_AC_ONLINE 1)\n"
    "set(VSS_PERF_FORBIDDEN_BATTERY_STATUS Discharging)\n"
    "set(VSS_PERF_INTEL_PSTATE_MODE active)\nset(VSS_PERF_SCALING_GOVERNOR powersave)\n"
    "set(VSS_PERF_INTEL_PSTATE_NO_TURBO 0)\nset(VSS_PERF_MAX_LOAD_1M 0.50)\n"
    "set(VSS_PERF_MAX_PACKAGE_TEMP_MILLIC 55000)\nset(VSS_PERF_STABLE_WINDOW_SECONDS 60)\n"
    "set(VSS_PERF_SAMPLE_INTERVAL_SECONDS 10)\nset(VSS_PERF_MAX_WAIT_SECONDS 900)\n"
    "set(VSS_PERF_TASKSET_PATH /usr/bin/taskset)\n")
endfunction()

function(make_fixture name)
  set(root "${work}/${name}/host")
  file(MAKE_DIRECTORY "${root}/proc")
  file(MAKE_DIRECTORY "${root}/etc")
  file(MAKE_DIRECTORY "${root}/sys/class/power_supply/AC")
  file(MAKE_DIRECTORY "${root}/sys/class/power_supply/BAT0")
  file(MAKE_DIRECTORY "${root}/sys/devices/system/cpu/intel_pstate")
  file(MAKE_DIRECTORY "${root}/sys/devices/system/cpu/cpu4/topology")
  foreach(cpu RANGE 0 7)
    file(MAKE_DIRECTORY "${root}/sys/devices/system/cpu/cpu${cpu}/cpufreq")
    file(WRITE "${root}/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor" "powersave\n")
  endforeach()
  file(MAKE_DIRECTORY "${root}/sys/devices/virtual/thermal/thermal_zone0")
  file(MAKE_DIRECTORY "${root}/sys/class/thermal")
  file(CREATE_LINK "${root}/sys/devices/virtual/thermal/thermal_zone0" "${root}/sys/class/thermal/thermal_zone0" SYMBOLIC)
  file(WRITE "${root}/proc/loadavg" "0.50 0.00 0.00 1/100 123\n")
  file(WRITE "${root}/proc/cpuinfo"
    "processor : 0\nmodel name : Synthetic test processor\n"
    "processor : 4\nmodel name : Synthetic test processor\n")
  file(WRITE "${root}/sys/devices/system/cpu/online" "0-7\n")
  file(WRITE "${root}/sys/devices/system/cpu/cpu4/topology/thread_siblings_list" "0,4\n")
  file(WRITE "${root}/sys/class/power_supply/AC/online" "1\n")
  file(WRITE "${root}/sys/class/power_supply/BAT0/status" "Charging\n")
  file(WRITE "${root}/sys/devices/system/cpu/intel_pstate/status" "active\n")
  file(WRITE "${root}/sys/devices/system/cpu/intel_pstate/no_turbo" "0\n")
  file(WRITE "${root}/sys/class/thermal/thermal_zone0/type" "x86_pkg_temp\n")
  file(WRITE "${root}/sys/class/thermal/thermal_zone0/temp" "55000\n")
  file(WRITE "${root}/etc/machine-id"
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n")
  set(taskset "${root}/usr/bin/taskset")
  file(MAKE_DIRECTORY "${root}/usr/bin")
  file(WRITE "${taskset}" "#!/bin/sh\nexit 0\n")
  file(CHMOD "${taskset}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
  write_profile("${work}/${name}/profile.cmake")
  file(WRITE "${work}/${name}/benchmark" "synthetic benchmark\n")
  file(SHA256 "${work}/${name}/benchmark" benchmark_sha)
  set("${name}_root" "${root}" PARENT_SCOPE)
  set("${name}_profile" "${work}/${name}/profile.cmake" PARENT_SCOPE)
  set("${name}_benchmark" "${work}/${name}/benchmark" PARENT_SCOPE)
  set("${name}_benchmark_sha" "${benchmark_sha}" PARENT_SCOPE)
  set("${name}_taskset" "${taskset}" PARENT_SCOPE)
endfunction()

function(run_case name trace expected_status expected_token)
  if(NOT DEFINED ${name}_root)
    make_fixture("${name}")
  endif()
  set(output "${work}/${name}/preflight.env")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DVSS_PERF_HOST_PROFILE_FILE=${${name}_profile}"
      "-DVSS_PERF_PREFLIGHT_OUTPUT=${output}"
      "-DVSS_PERF_SOURCE_DIR=${source_dir}"
      "-DVSS_PERF_EXPECTED_COMMIT=${expected_commit}"
      "-DVSS_PERF_BENCHMARK=${${name}_benchmark}"
      "-DVSS_PERF_EXPECTED_BENCHMARK_SHA256=${${name}_benchmark_sha}"
      "-DVSS_PERF_RECORD_PROFILE_ID=${profile_id}"
      "-DVSS_PERF_CONFIGURATION=${configuration}"
      "-DVSS_PERF_HOST_ROOT=${${name}_root}"
      "-DVSS_PERF_TASKSET=${${name}_taskset}"
      "-DVSS_PERF_TEST_SAMPLE_TRACE=${trace}"
      -P "${checker}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output_text ERROR_VARIABLE error_text)
  if("${expected_status}" STREQUAL "PASS" AND NOT status EQUAL 0)
    message(FATAL_ERROR "${name}: expected preflight pass, got ${status}\n${output_text}${error_text}")
  endif()
  if("${expected_status}" STREQUAL "FAIL" AND status EQUAL 0)
    message(FATAL_ERROR "${name}: expected fail-closed preflight")
  endif()
  if(NOT EXISTS "${output}")
    message(FATAL_ERROR "${name}: preflight output was not written")
  endif()
  if(EXISTS "${output}.tmp")
    message(FATAL_ERROR "${name}: temporary preflight output remains after publication")
  endif()
  file(READ "${output}" result)
  string(REGEX MATCH "(^|\n)RESULT=[^\n]+" result_line "${result}")
  if(NOT result_line MATCHES "RESULT=${expected_token}")
    message(FATAL_ERROR "${name}: expected ${expected_token}, got ${result_line}")
  endif()
endfunction()

run_case(boundary "${trace_pass}" PASS preflight_pass)
run_case(load-boundary "${trace_high_load}" FAIL preflight_failed)
run_case(middle-reset "${trace_reset}" FAIL preflight_failed)
run_case(timeout "${trace_timeout}" FAIL preflight_failed)
run_case(temperature-boundary "${trace_hot}" FAIL preflight_failed)

make_fixture(ac-off)
file(WRITE "${ac-off_root}/sys/class/power_supply/AC/online" "0\n")
run_case(ac-off "${trace_pass}" FAIL preflight_failed)

make_fixture(battery-offline)
file(WRITE "${battery-offline_root}/sys/class/power_supply/BAT0/status" "Discharging\n")
run_case(battery-offline "${trace_pass}" FAIL preflight_failed)

make_fixture(missing-thermal)
file(REMOVE "${missing-thermal_root}/sys/class/thermal/thermal_zone0/type")
run_case(missing-thermal "${trace_pass}" FAIL preflight_failed)

make_fixture(cpu-mismatch)
file(WRITE "${cpu-mismatch_root}/proc/cpuinfo"
  "processor : 0\nmodel name : different-cpu\n")
run_case(cpu-mismatch "${trace_pass}" FAIL preflight_failed)

make_fixture(governor-mismatch)
file(WRITE "${governor-mismatch_root}/sys/devices/system/cpu/cpu3/cpufreq/scaling_governor" "performance\n")
run_case(governor-mismatch "${trace_pass}" FAIL preflight_failed)

make_fixture(missing-taskset)
file(REMOVE "${missing-taskset_taskset}")
run_case(missing-taskset "${trace_pass}" FAIL preflight_failed)

make_fixture(wrong-commit)
make_fixture(machine-id-32)
file(WRITE "${machine-id-32_root}/etc/machine-id"
  "0123456789abcdef0123456789abcdef\n")
run_case(machine-id-32 "${trace_pass}" PASS preflight_pass)
set(machine_id_32_value "0123456789abcdef0123456789abcdef")
string(SHA256 expected_machine_id_32_hash "${machine_id_32_value}")
file(READ "${work}/machine-id-32/preflight.env" machine_id_32_output)
if(NOT machine_id_32_output MATCHES
   "HOST_FINGERPRINT_SHA256=${expected_machine_id_32_hash}")
  message(FATAL_ERROR
    "machine-id-32: preflight did not persist the expected host fingerprint")
endif()
if(machine_id_32_output MATCHES "${machine_id_32_value}")
  message(FATAL_ERROR "machine-id-32: raw machine identifier leaked")
endif()

make_fixture(machine-id-short)
file(WRITE "${machine-id-short_root}/etc/machine-id" "0123456789abcdef\n")
run_case(machine-id-short "${trace_pass}" FAIL preflight_failed)

set(saved_expected_commit "${expected_commit}")
set(expected_commit 0000000000000000000000000000000000000000)
run_case(wrong-commit "${trace_pass}" FAIL preflight_failed)
set(expected_commit "${saved_expected_commit}")
message(STATUS "local capture environment policy RED tests executed")
