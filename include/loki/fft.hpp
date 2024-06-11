#pragma once

#include <complex>
#include <cstddef>
#include <span>

#include <fftw3.h>
#include <stdexcept>

using Complex = std::complex<float>;

class FFTW {
public:
    FFTW(size_t n, int sign);
    FFTW(const FFTW&)            = delete;
    FFTW& operator=(const FFTW&) = delete;
    FFTW(FFTW&&)                 = delete;
    FFTW& operator=(FFTW&&)      = delete;
    virtual ~FFTW();

    void fft(std::span<const Complex> in, std::span<Complex> out);
    void fft(std::span<const float> in, std::span<Complex> out = {});
    void fft(std::span<const Complex> in, std::span<float> out);

    template <typename T>
    void shift(std::span<T> arr, size_t nx, size_t ny, size_t nz = 1) {
        if (nx % 2 != 0 || ny % 2 != 0) {
            throw std::runtime_error(
                "Shift is not implemented for odd nx or ny");
        }

        const size_t nyp  = (nz == 1) ? (ny / 2 + 1) : ny;
        const size_t nzp  = (nz == 1) ? 1 : (nz / 2 + 1);
        const size_t nyzp = nyp * nzp;

        for (size_t i = 0; i < nx; i++) {
            for (size_t j = (i % 2 != 0 ? nzp : 0); j < nyzp; j += 2 * nzp) {
                for (size_t k = 0; k < nzp; k++) {
                    if constexpr (std::is_same_v<T, Complex>) {
                        arr[i * nyzp + j + k] = -arr[i * nyzp + j + k];
                    } else {
                        arr[i * nyp * nz + j * nz + k] =
                            -arr[i * nyp * nz + j * nz + k];
                    }
                }
            }
        }
    }

    // In-place shift of Fourier origin to (nx/2,0) for even nx.
    static void shift(std::span<Complex> arr, size_t nx, size_t ny) {
        size_t nyp  = ny / 2 + 1;
        size_t stop = nx * nyp;
        if (nx % 2 != 0) {
            throw std::runtime_error("Shift is not implemented for odd nx");
        }
        size_t inc = 2 * nyp;
        for (size_t i = nyp; i < stop; i += inc) {
            for (size_t j = 0; j < nyp; j++) {
                arr[i + j] = -arr[i + j];
            }
        }
    }

    // Out-of-place shift of Fourier origin to (nx/2,0) for even nx.
    static void shift(std::span<float> arr, size_t nx, size_t ny) {
        if (nx % 2 != 0) {
            throw std::runtime_error("Shift is not implemented for odd nx");
        }
        size_t stop = nx * ny;
        size_t inc  = 2 * ny;
        for (size_t i = ny; i < stop; i += inc) {
            for (size_t j = 0; j < ny; j++) {
                arr[i + j] = -arr[i + j];
            }
        }
    }

    // In-place shift of Fourier origin to (nx/2,ny/2,0) for even nx and ny.
    static void shift(std::span<Complex> arr, size_t nx, size_t ny, size_t nz) {
        size_t nzp  = nz / 2 + 1;
        size_t nyzp = ny * nzp;
        if (nx % 2 != 0 || ny % 2 != 0) {
            throw std::runtime_error(
                "Shift is not implemented for odd nx or ny");
        }
        size_t pinc = 2 * nzp;
        for (size_t i = 0; i < nx; i++) {
            bool is_odd_row = (i % 2 != 0);
            for (size_t j = (is_odd_row ? nzp : 0); j < nyzp; j += pinc) {
                for (size_t k = 0; k < nzp; k++) {
                    arr[i * nyzp + j + k] = -arr[i * nyzp + j + k];
                }
            }
        }
    }

    // Out-of-place shift of Fourier origin to (nx/2,ny/2,0) for even nx and ny.
    static void shift(std::span<float> arr, size_t nx, size_t ny, size_t nz) {
        size_t nyz = ny * nz;
        if (nx % 2 != 0 || ny % 2 != 0) {
            throw std::runtime_error(
                "Shift is not implemented for odd nx or ny");
        }
        size_t pinc = 2 * nz;
        for (size_t i = 0; i < nx; i++) {
            bool is_odd_row = (i % 2 != 0);
            for (size_t j = (is_odd_row ? nz : 0); j < nyz; j += pinc) {
                for (size_t k = 0; k < nz; k++) {
                    arr[i * nyz + j + k] = -arr[i * nyz + j + k];
                }
            }
        }
    }

protected:
    float m_norm;
    int m_sign;
    fftwf_plan m_plan;

    virtual void plan(std::span<Complex> in, std::span<Complex> out);
    virtual void plan(std::span<float> in, std::span<Complex> out);
    virtual void plan(std::span<Complex> in, std::span<float> out);
    virtual void execute(std::span<Complex> in, std::span<Complex> out);
    virtual void execute(std::span<float> in, std::span<Complex> out);
    virtual void execute(std::span<Complex> in, std::span<float> out);
    void setup(std::span<Complex> in, std::span<Complex> out);
    void setup(std::span<float> in, std::span<Complex> out);
    void setup(std::span<Complex> in, std::span<float> out);
};

class FFT1D : public FFTW {
public:
    explicit FFT1D(size_t nx,
                   int sign,
                   std::span<Complex> in  = {},
                   std::span<Complex> out = {});

    void plan(std::span<Complex> in, std::span<Complex> out) override;
    void execute(std::span<Complex> in, std::span<Complex> out) override;

private:
    size_t m_nx;
};

class FFT1DR2C : public FFTW {
public:
    explicit FFT1DR2C(size_t nx,
                      std::span<float> in    = {},
                      std::span<Complex> out = {});

    void plan(std::span<float> in, std::span<Complex> out) override;
    void execute(std::span<float> in, std::span<Complex> out) override;

private:
    size_t m_nx;
};

class FFT1DC2R : public FFTW {
public:
    explicit FFT1DC2R(size_t nx,
                      std::span<Complex> in = {},
                      std::span<float> out  = {});

    void plan(std::span<Complex> in, std::span<float> out) override;
    void execute(std::span<Complex> in, std::span<float> out) override;

private:
    size_t m_nx;
};

class FFT2D : public FFTW {
public:
    FFT2D(size_t nx,
          size_t ny,
          int sign,
          std::span<Complex> in  = {},
          std::span<Complex> out = {});

    void plan(std::span<Complex> in, std::span<Complex> out) override;
    void execute(std::span<Complex> in, std::span<Complex> out) override;

private:
    size_t m_nx;
    size_t m_ny;
};

class FFT2DR2C : public FFTW {
public:
    FFT2DR2C(size_t nx,
             size_t ny,
             std::span<float> in    = {},
             std::span<Complex> out = {});

    void plan(std::span<float> in, std::span<Complex> out) override;
    void execute(std::span<float> in, std::span<Complex> out) override;

private:
    size_t m_nx;
    size_t m_ny;
};

class FFT2DC2R : public FFTW {
public:
    explicit FFT2DC2R(size_t nx,
                      size_t ny,
                      std::span<Complex> in = {},
                      std::span<float> out  = {});

    void plan(std::span<Complex> in, std::span<float> out) override;
    void execute(std::span<Complex> in, std::span<float> out) override;

private:
    size_t m_nx;
    size_t m_ny;
};

class FFT2DC {
public:
    FFT2DC(int n1x, int n2x, int ny);
    FFT2DC(const FFT2DC&)            = delete;
    FFT2DC& operator=(const FFT2DC&) = delete;
    FFT2DC(FFT2DC&&)                 = delete;
    FFT2DC& operator=(FFT2DC&&)      = delete;
    ~FFT2DC();

    void circular_convolve(std::span<float> n1,
                           std::span<float> n2,
                           std::span<float> out);

private:
    int m_n1x;
    int m_n2x;
    int m_ny;

    size_t m_fft_size;
    fftwf_complex* m_n1_fft;
    fftwf_complex* m_n2_fft;
    fftwf_complex* m_n1n2_fft;
    fftwf_plan m_plan_fw;
    fftwf_plan m_plan_bw;
};