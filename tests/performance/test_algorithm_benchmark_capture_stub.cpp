#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
  std::string output;
  bool plan_seen = false;
  bool digest_seen = false;
  bool governor_seen = false;
  bool turbo_seen = false;
  bool runner_seen = false;
  const std::span<char*> arguments(argv, static_cast<std::size_t>(argc));
  std::size_t index = 1;
  while (index < arguments.size()) {
    const std::string_view argument(arguments[index]);
    if (index + 1 >= arguments.size()) {
      return 20;
    }
    const std::string_view value(arguments[index + 1]);
    if (argument == "--output") {
      output = value;
    } else if (argument == "--batch-plan") {
      plan_seen = value.find("plan with spaces.env") != std::string_view::npos;
    } else if (argument == "--benchmark-sha256") {
      digest_seen = value.size() == 64U;
    } else if (argument == "--frequency-governor") {
      governor_seen = value == "performance";
    } else if (argument == "--turbo-state") {
      turbo_seen = value == "enabled";
    } else if (argument == "--runner-label") {
      runner_seen = value == "wrapper-test";
    } else {
      return 21;
    }
    index += 2U;
  }
  if (output.empty() || !plan_seen || !digest_seen || !governor_seen ||
      !turbo_seen || !runner_seen) {
    std::cerr << "wrapper arguments were not forwarded exactly\n";
    return 22;
  }
  std::cout << "stub stdout diagnostic\n";
  const char* const failure = std::getenv("VSS_TEST_BENCHMARK_FAILURE");
  if (failure != nullptr && std::string_view(failure) == "2") {
    std::cerr << "stub allocation counter failure\n";
    return 2;
  }
  std::ofstream record(output, std::ios::binary | std::ios::trunc);
  const char* const fixturePath = std::getenv("VSS_TEST_BENCHMARK_RECORD");
  if (fixturePath == nullptr || *fixturePath == '\0') {
    record << "{\"schema_version\":2}\n";
  } else {
    const std::ifstream fixture(fixturePath, std::ios::binary);
    if (!fixture) {
      std::cerr << "test record fixture could not be opened\n";
      return 24;
    }
    record << fixture.rdbuf();
  }
  return record ? 0 : 23;
}
