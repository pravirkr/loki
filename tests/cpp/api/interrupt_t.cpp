#include "loki/common/interrupt.hpp"

#include <csignal>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <signal.h> // NOLINT(modernize-deprecated-headers): POSIX sigaction
#include <sys/wait.h>
#include <unistd.h>

#include "loki/common/backend.hpp"
#include "loki/detection/thresholds.hpp"

namespace {

// The handler currently installed for sig.
void (*current_handler(int sig))(int) {
    struct sigaction action{};
    sigaction(sig, nullptr, &action);
    return action.sa_handler;
}

} // namespace

TEST_CASE("interrupt requests are checked and cleared", "[interrupt]") {
    loki::clear_interrupt();
    REQUIRE_FALSE(loki::interrupt_requested());
    REQUIRE_NOTHROW(loki::check_interrupt());
    loki::request_interrupt();
    REQUIRE(loki::interrupt_requested());
    REQUIRE(loki::interrupt_signal() == 0);
    REQUIRE_THROWS_AS(loki::check_interrupt(), loki::Interrupted);
    loki::clear_interrupt();
    REQUIRE_NOTHROW(loki::check_interrupt());
}

TEST_CASE("a guard turns SIGINT into a request and restores the handler",
          "[interrupt]") {
    const auto before = current_handler(SIGINT);
    {
        const loki::InterruptGuard outer;
        {
            const loki::InterruptGuard inner; // nested: no reinstall
            REQUIRE(std::raise(SIGINT) == 0);
            // Still alive: the signal became a request.
            REQUIRE(loki::interrupt_requested());
            REQUIRE(loki::interrupt_signal() == SIGINT);
        }
        // The inner guard leaves the request to the outer one.
        REQUIRE(loki::interrupt_requested());
        REQUIRE(current_handler(SIGINT) != before);
    }
    REQUIRE_FALSE(loki::interrupt_requested());
    REQUIRE(current_handler(SIGINT) == before);
}

TEST_CASE("SIGTERM is routed only on request", "[interrupt]") {
    const auto before = current_handler(SIGTERM);
    {
        const loki::InterruptGuard guard;
        REQUIRE(current_handler(SIGTERM) == before);
    }
    {
        const loki::InterruptGuard guard(/*handle_terminate=*/true);
        REQUIRE(current_handler(SIGTERM) != before);
    }
    REQUIRE(current_handler(SIGTERM) == before);
}

TEST_CASE("a second SIGINT with a default handler kills the process",
          "[interrupt]") {
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        std::signal(SIGINT, SIG_DFL);
        const loki::InterruptGuard guard;
        std::raise(SIGINT); // a request
        std::raise(SIGINT); // the default action: terminate
        _exit(0);
    }
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFSIGNALED(status));
    REQUIRE(WTERMSIG(status) == SIGINT);
}

TEST_CASE("a pending request stops DynamicThresholdScheme::run",
          "[interrupt][thresholds]") {
    const auto* mode = GENERATE("legacy", "shared");
    CAPTURE(mode);
    const std::vector<float> branching_pattern(4, 2.0F);
    loki::detection::DynamicThresholdScheme scheme(
        branching_pattern, 0.1F, 32, 128, 8, 0.1F, 6.0F, 20, 0.3F, 1.0F, 1.5F,
        0, mode, /*seed=*/7, /*batch_size=*/256, loki::Exec::cpu(1));
    loki::request_interrupt();
    REQUIRE_THROWS_AS(scheme.run(10, /*show_progress=*/false),
                      loki::Interrupted);
    loki::clear_interrupt();
    // Usable again once the request is cleared.
    REQUIRE_NOTHROW(scheme.run(10, /*show_progress=*/false));
}
