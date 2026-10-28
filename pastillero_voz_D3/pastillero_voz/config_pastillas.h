// Del entrenamiento (export/config_dispensador.json). NO editar: si reentrenas se reemplaza.
#pragma once
#define FS 16000
#define N_MUESTRAS 16000
#define FRAME_LEN 255
#define FRAME_STEP 128
#define FFT_LEN 256
#define N_FRAMES 124
#define N_BINS 129
#define N_CLASES 5

const float NORM_MU = 0.254228294f;
const float NORM_SD = 0.480546921f;
const float ENTRADA_ESCALA = 0.0381736644f;
const int   ENTRADA_ZP = -114;
const float SALIDA_ESCALA = 0.00390625f;
const int   SALIDA_ZP = -128;

const float RMS_MIN = 0.02578f;    // por debajo de esto se considera silencio

const char* const CLASES[N_CLASES] = {"dolex", "advil", "aspirina", "omeprazol", "losartan"};
