#ifndef TEMPERATURECALCULATOR_H
#define TEMPERATURECALCULATOR_H

#include <cstdint>
#include <vector>

#if defined(_WIN32)
    #if defined(TEMPERATURE_CALCULATOR_BUILD_DLL)
        #define TEMPERATURE_CALCULATOR_API __declspec(dllexport)
    #else
        #define TEMPERATURE_CALCULATOR_API __declspec(dllimport)
    #endif
#else
    #define TEMPERATURE_CALCULATOR_API __attribute__((visibility("default")))
#endif

/**
 * @brief Temperature calculator for radiometric conversion.
 *
 * Stores radiometric coefficients read once from the device and
 * computes Celsius temperature from per-query runtime inputs and raw pixel data.
 */
class CTC_TemperatureCalculator
{
public:
    TEMPERATURE_CALCULATOR_API CTC_TemperatureCalculator();
    TEMPERATURE_CALCULATOR_API ~CTC_TemperatureCalculator();

    /**
     * @brief Prints a simple runtime status message for library health check.
     */
    TEMPERATURE_CALCULATOR_API void CTC_PrintLibraryStatus() const;

    /**
     * @brief Stores radiometric parameters used by the temperature formula.
     *
     * Expected parameter count is 5:
     * [0] board-temp correction coefficient
     * [1] shutter-temp offset coefficient
     * [2] PWM correction coefficient
     * [3] PWM denominator
     * [4] reserved (kept for compatibility with current device block)
     *
     * @param params Vector of radiometric parameters read from the device.
     * @throws std::invalid_argument if params contains fewer than 5 elements.
     */
    TEMPERATURE_CALCULATOR_API void CTC_SetParameter(const std::vector<float>& params);

    /**
     * @brief Computes temperature in Celsius from raw data and dynamic inputs.
     *
     * @param rawValue        Raw signed 16-bit value from PIXEL_DATA.
     * @param boardTempC      Board temperature in Celsius.
     * @param radiometricGain Radiometric gain read from the device register.
     * @param pwmTarget       PWM target read from the device register.
     * @return Temperature in Celsius.
     * @throws std::runtime_error if parameters are uninitialized or invalid.
     */
    TEMPERATURE_CALCULATOR_API double CTC_ComputeTemperature(int16_t rawValue,
                                                         float boardTempC,
                                                         float radiometricGain,
                                                         float pwmTarget) const;

private:
    std::vector<float> m_params;
};

// ---------------------------------------------------------------------------
// C API for Python ctypes (and other FFI consumers)
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

typedef void* TempCalcHandle;

TEMPERATURE_CALCULATOR_API TempCalcHandle CTC_Create(void);
TEMPERATURE_CALCULATOR_API void           CTC_Destroy(TempCalcHandle h);
TEMPERATURE_CALCULATOR_API void           CTC_PrintLibraryStatus(TempCalcHandle h);
TEMPERATURE_CALCULATOR_API int            CTC_SetParameter(TempCalcHandle h,
                                                           const float* params, int count);
TEMPERATURE_CALCULATOR_API int            CTC_ComputeTemperature(TempCalcHandle h,
                                                                  int16_t raw,
                                                                  float boardTempC,
                                                                  float radiometricGain,
                                                                  float pwmTarget,
                                                                  double* out);

#ifdef __cplusplus
}
#endif

#endif /* TEMPERATURECALCULATOR_H */
