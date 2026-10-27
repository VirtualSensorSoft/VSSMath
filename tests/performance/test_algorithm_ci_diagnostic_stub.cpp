#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string_view>

namespace {

const char* environment(const char* name) { return std::getenv(name); }

bool readInvocationCount(const std::filesystem::path& path, int& count) {
  if (!std::filesystem::exists(path)) {
    count = 0;
    return true;
  }
  std::ifstream input(path);
  return static_cast<bool>(input >> count) && count >= 0;
}

} // namespace

int main(int argc, char** argv) {
  std::filesystem::path output;
  bool digest_seen = false;
  bool governor_seen = false;
  bool turbo_seen = false;
  bool runner_seen = false;
  const std::span<char*> arguments(argv, static_cast<std::size_t>(argc));
  std::size_t index = 1;
  while (index < arguments.size()) {
    if (index + 1U >= arguments.size()) {
      return 20;
    }
    const std::string_view option(arguments[index]);
    const std::string_view value(arguments[index + 1U]);
    if (option == "--output") {
      output = value;
    } else if (option == "--benchmark-sha256") {
      digest_seen = value.size() == 64U;
    } else if (option == "--frequency-governor") {
      governor_seen = value == "performance";
    } else if (option == "--turbo-state") {
      turbo_seen = value == "enabled";
    } else if (option == "--runner-label") {
      runner_seen = value == "performance-ci-diagnostic-test";
    } else {
      return 21;
    }
    index += 2U;
  }
  const char* const fixture_directory = environment("VSS_TEST_FIXTURE_DIR");
  const char* const invocation_file = environment("VSS_TEST_INVOCATION_FILE");
  const char* const mode = environment("VSS_TEST_DIAGNOSTIC_MODE");
  if (output.empty() || !digest_seen || !governor_seen || !turbo_seen ||
      !runner_seen || fixture_directory == nullptr ||
      invocation_file == nullptr || mode == nullptr) {
    std::cerr << "diagnostic stub configuration or arguments are incomplete\n";
    return 22;
  }

  int invocation = 0;
  const std::filesystem::path invocation_path(invocation_file);
  if (!readInvocationCount(invocation_path, invocation)) {
    std::cerr << "could not read the invocation counter\n";
    return 23;
  }
  {
    std::ofstream count_output(invocation_path, std::ios::trunc);
    count_output << invocation + 1 << '\n';
    if (!count_output) {
      return 24;
    }
  }

  std::cout << "diagnostic capture " << invocation + 1 << " stdout\n";
  std::cerr << "diagnostic capture " << invocation + 1 << " stderr\n";
  if (std::string_view(mode) == "missing-second" && invocation == 1) {
    return 0;
  }

  const std::filesystem::path source =
      std::filesystem::path(fixture_directory) /
      (invocation == 0 ? "first.json" : "second.json");
  const std::ifstream input(source, std::ios::binary);
  std::ofstream output_stream(output, std::ios::binary | std::ios::trunc);
  output_stream << input.rdbuf();
  if (!input || !output_stream) {
    std::cerr << "could not transfer the synthetic record\n";
    return 25;
  }
  return 0;
}
