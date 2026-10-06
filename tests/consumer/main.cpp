// Uses only the installed public headers: one FFA workspace and FFT plan
// cache shared by two FFA instances, on the CPU backend.
#include <cstdio>
#include <vector>

#include "loki/loki.hpp"

int main() {
    using namespace loki;
    constexpr SizeType kNsamps = 1U << 14U;

    const std::vector<ParamLimit> limits = {{.min = 5.0, .max = 10.0}};
    const search::PulsarSearchConfig cfg(kNsamps, 1.0e-3, 32, 0.5, limits);

    const algorithms::FFA<float> owner(cfg, /*show_progress=*/false);
    memory::FFAWorkspace<float> workspace(owner.get_plan(), Exec::cpu());
    math::FFTManager fft_manager(Exec::cpu());

    const std::vector<float> ts_e(kNsamps, 0.0F);
    const std::vector<float> ts_v(kNsamps, 1.0F);
    for (int run = 0; run < 2; ++run) {
        algorithms::FFA<float> ffa(workspace, fft_manager, cfg,
                                   /*show_progress=*/false);
        std::vector<float> fold(ffa.get_plan().get_buffer_size());
        ffa.execute(ts_e, ts_v, fold);
    }

    std::printf("loki consumer OK; backends:");
    for (const auto backend : available_backends()) {
        std::printf(" %s", to_string(backend).data());
    }
    std::printf("\n");
    return 0;
}
