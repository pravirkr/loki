#include "lib/detail/progress.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <sys/wait.h>
#include <unistd.h>

#include "loki/common/progress.hpp"
#include "loki/common/types.hpp"

namespace {

using loki::ProgressMode;
using loki::SizeType;
using loki::progress::MultiprocessProgressTracker;
using loki::progress::ProgressBar;
using loki::progress::ProgressTracker;

// Everything written to stdout (the C stream) while fn runs.
template <typename Fn> std::string capture_stdout(Fn&& fn) {
    std::fflush(stdout);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): closed below
    std::FILE* tmp = std::tmpfile();
    REQUIRE(tmp != nullptr);
    const int saved = dup(STDOUT_FILENO);
    REQUIRE(saved >= 0);
    dup2(fileno(tmp), STDOUT_FILENO); // NOLINT(misc-include-cleaner): POSIX
    std::forward<Fn>(fn)();
    std::fflush(stdout);
    dup2(saved, STDOUT_FILENO);
    close(saved);
    std::string out;
    REQUIRE(std::fseek(tmp, 0, SEEK_SET) == 0);
    for (int c = std::fgetc(tmp); c != EOF; c = std::fgetc(tmp)) {
        out += static_cast<char>(c);
    }
    std::fclose(tmp); // NOLINT(cppcoreguidelines-owning-memory): see above
    return out;
}

// Restores the process-wide mode at the end of a test.
struct ModeScope {
    explicit ModeScope(ProgressMode mode) : m_prev(loki::get_progress_mode()) {
        loki::set_progress_mode(mode);
    }
    ~ModeScope() { loki::set_progress_mode(m_prev); }
    ModeScope(const ModeScope&)            = delete;
    ModeScope& operator=(const ModeScope&) = delete;
    ModeScope(ModeScope&&)                 = delete;
    ModeScope& operator=(ModeScope&&)      = delete;

private:
    ProgressMode m_prev;
};

void run_bar(bool show, SizeType n = 8) {
    ProgressTracker bar("Computing scheme", n, show);
    for (SizeType i = 1; i <= n; ++i) {
        bar.set_progress(i);
    }
    bar.mark_as_completed();
}

} // namespace

TEST_CASE("progress mode names", "[progress]") {
    for (const auto m : {ProgressMode::kAuto, ProgressMode::kRich,
                         ProgressMode::kPlain, ProgressMode::kOff}) {
        REQUIRE(loki::parse_progress_mode(loki::to_string(m)) == m);
    }
    REQUIRE_THROWS_AS(loki::parse_progress_mode("fancy"),
                      std::invalid_argument);
}

TEST_CASE("show_progress=false writes nothing in any mode", "[progress]") {
    for (const auto m : {ProgressMode::kAuto, ProgressMode::kRich,
                         ProgressMode::kPlain, ProgressMode::kOff}) {
        const ModeScope scope(m);
        REQUIRE(capture_stdout([] { run_bar(/*show=*/false); }).empty());
        // A bar destroyed without an explicit completion, too.
        REQUIRE(capture_stdout([] {
                    const ProgressTracker bar("x", 4, /*show=*/false);
                }).empty());
    }
}

TEST_CASE("off mode writes nothing", "[progress]") {
    const ModeScope scope(ProgressMode::kOff);
    REQUIRE(capture_stdout([] { run_bar(/*show=*/true); }).empty());
}

TEST_CASE("plain mode prints quarters and a final line", "[progress]") {
    const ModeScope scope(ProgressMode::kPlain);
    const auto out = capture_stdout([] { run_bar(/*show=*/true, 8); });
    REQUIRE(out.find('\033') == std::string::npos);
    REQUIRE(out.find('\r') == std::string::npos);
    // 25 %, 50 %, 75 % and the completed line.
    REQUIRE(std::ranges::count(out, '\n') == 4);
    REQUIRE(out.find(" 25%") != std::string::npos);
    REQUIRE(out.find("100%") != std::string::npos);
    REQUIRE(out.starts_with("Computing scheme"));
}

TEST_CASE("plain mode with several tasks prints the overall count",
          "[progress]") {
    const ModeScope scope(ProgressMode::kPlain);
    const auto out = capture_stdout([] {
        MultiprocessProgressTracker tracker("Pruning tree");
        tracker.start();
        for (int t = 0; t < 4; ++t) {
            const int id = tracker.add_task("Pruning segment", 10);
            ProgressTracker task(&tracker, id);
            task.set_progress(5);
            task.set_progress(10);
        }
        tracker.stop();
    });
    REQUIRE(out.find('\033') == std::string::npos);
    REQUIRE(out.find("Pruning segment") == std::string::npos);
    REQUIRE(out.find("4/4") != std::string::npos);
}

TEST_CASE("rich mode leaves the terminal as it found it", "[progress]") {
    const ModeScope scope(ProgressMode::kRich);
    const auto out = capture_stdout([] { run_bar(/*show=*/true); });
    // Cursor hidden and line wrapping off while live, both restored last;
    // a transient bar leaves nothing behind it.
    REQUIRE(out.starts_with("\033[?25l\033[?7l"));
    REQUIRE(out.ends_with("\033[?7h\033[?25h"));
}

TEST_CASE("a non-transient bar stays after completion", "[progress]") {
    const ModeScope scope(ProgressMode::kRich);
    const auto out  = capture_stdout([] {
        ProgressTracker bar("Kept bar", 2, /*show=*/true,
                            ProgressBar::Kind::kStandard,
                            /*transient=*/false);
        bar.set_progress(2);
        bar.mark_as_completed();
    });
    const auto kept = out.rfind("Kept bar");
    REQUIRE(kept != std::string::npos);
    // Its line ends with a newline, and no cursor-up follows it.
    const auto tail = std::string_view(out).substr(kept);
    REQUIRE(tail.find('\n') != std::string_view::npos);
    REQUIRE(tail.find('A') == std::string_view::npos);
}

TEST_CASE("a completed bar keeps its elapsed time", "[progress]") {
    ProgressBar bar("x", 2);
    bar.set_progress(2);
    REQUIRE(bar.is_completed());
    const auto elapsed = bar.get_elapsed();
    REQUIRE(elapsed.count() > 0);
    REQUIRE(bar.get_elapsed() == elapsed);
}

TEST_CASE("a bar shrinks to fit the terminal width", "[progress]") {
    const ProgressBar bar("A rather long prefix for a bar", 10,
                          ProgressBar::Kind::kPruning);
    const loki::progress::RenderContext ctx{};
    const auto wide   = bar.to_string(ctx);
    const auto narrow = bar.to_string(ctx, 70);
    REQUIRE(narrow.size() < wide.size());
}

TEST_CASE("rich multi-task display: logs above, no jump at the end",
          "[progress]") {
    const ModeScope scope(ProgressMode::kRich);
    const auto out = capture_stdout([] {
        MultiprocessProgressTracker tracker("Pruning tree");
        std::vector<int> ids;
        ids.reserve(3);
        for (int t = 0; t < 3; ++t) {
            ids.push_back(tracker.add_task("Pruning segment", 10));
        }
        tracker.start();
        {
            ProgressTracker early(&tracker, ids[0]);
            ProgressTracker late1(&tracker, ids[1]);
            ProgressTracker late2(&tracker, ids[2]);
            for (SizeType i = 1; i <= 5; ++i) {
                early.set_progress(2 * i);
                late1.set_progress(i);
                late2.set_progress(i);
                std::this_thread::sleep_for(std::chrono::milliseconds(60));
            }
            // What the display's spdlog sink does with a log line.
            tracker.queue_log("mid-run message\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        tracker.stop();
    });
    REQUIRE(out.find("Pruning tree") != std::string::npos);
    const auto log_pos = out.find("mid-run message");
    REQUIRE(log_pos != std::string::npos);
    // The log line is printed with wrapping on, between region redraws.
    const auto wrap_on = out.rfind("\033[?7h", log_pos);
    REQUIRE(wrap_on != std::string::npos);
    REQUIRE(out.find("\033[?7l", log_pos) != std::string::npos);
    // Every cursor-up starts a redraw: \r before it, erase-below after it.
    bool saw_cursor_up = false;
    for (auto pos = out.find("\033["); pos != std::string::npos;
         pos      = out.find("\033[", pos + 1)) {
        auto end = pos + 2;
        while (end < out.size() && out[end] >= '0' && out[end] <= '9') {
            ++end;
        }
        if (end > pos + 2 && end < out.size() && out[end] == 'A') {
            saw_cursor_up = true;
            REQUIRE(out[pos - 1] == '\r');
            REQUIRE(std::string_view(out).substr(end + 1, 3) == "\033[J");
        }
    }
    REQUIRE(saw_cursor_up); // the region had several lines
    // Nothing is redrawn after the last erase: the region is cleared and the
    // terminal restored.
    const auto last_erase = out.rfind("\033[J");
    REQUIRE(last_erase != std::string::npos);
    REQUIRE(std::string_view(out).substr(last_erase) ==
            "\033[J\033[?7h\033[?25h");
}

TEST_CASE("a bar unwound by an exception prints no completion line",
          "[progress]") {
    const ModeScope scope(ProgressMode::kPlain);
    const auto out = capture_stdout([] {
        try {
            ProgressTracker bar("Stopped bar", 8, /*show=*/true);
            bar.set_progress(1);
            throw std::runtime_error("stop");
        } catch (const std::runtime_error&) { // NOLINT(bugprone-empty-catch)
        }
    });
    REQUIRE(out.empty());
}

TEST_CASE("a fatal signal during a live bar restores the terminal",
          "[progress]") {
    std::array<int, 2> fds{};
    REQUIRE(pipe(fds.data()) == 0);
    const pid_t pid = fork();
    REQUIRE(pid >= 0);
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        std::signal(SIGINT, SIG_DFL);
        loki::set_progress_mode(ProgressMode::kRich);
        ProgressTracker bar("Live bar", 8, /*show=*/true);
        bar.set_progress(2);
        std::raise(SIGINT);
        _exit(0);
    }
    close(fds[1]);
    std::string out;
    std::array<char, 4096> buf{};
    for (auto n = read(fds[0], buf.data(), buf.size()); n > 0;
         n      = read(fds[0], buf.data(), buf.size())) {
        out.append(buf.data(), static_cast<std::size_t>(n));
    }
    close(fds[0]);
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    REQUIRE(WIFSIGNALED(status));
    REQUIRE(WTERMSIG(status) == SIGINT);
    REQUIRE(out.starts_with("\033[?25l\033[?7l"));
    // The terminal ends restored: the restore comes after the last hide
    // cursor / wrap off. (A frame the render thread was writing at that
    // instant may still follow it; that does not change the terminal state.)
    const auto restore = out.rfind("\033[?7h\033[?25h");
    REQUIRE(restore != std::string::npos);
    CHECK(out.find("\033[?25l", restore) == std::string::npos);
    CHECK(out.find("\033[?7l", restore) == std::string::npos);
}
