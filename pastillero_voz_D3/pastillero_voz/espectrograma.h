// espectrograma.h
// Convierte 1 s de audio (16000 muestras int16) en la entrada int8 del modelo,
// haciendo EXACTAMENTE lo mismo que Colab:
//   audio/32768 -> STFT (ventana Hann de 255 como TensorFlow, paso 128, FFT de 256)
//   -> |magnitud| -> log1p -> (x - mu) / sd -> cuantizar a int8.
// Solo usa C++ estándar, sin librerías externas.
#pragma once
#include <math.h>
#include <stdint.h>
#include "config_pastillas.h"

namespace espectro {

static float ventana[FRAME_LEN];
static float cos_tab[FFT_LEN / 2], sin_tab[FFT_LEN / 2];
static uint16_t bit_rev[FFT_LEN];
static float re[FFT_LEN], im[FFT_LEN];

inline void iniciar() {
  const float PI2 = 6.283185307179586f;
  // Igual que tf.signal.hann_window(FRAME_LEN, periodic=True): con largo IMPAR (255)
  // TensorFlow divide entre FRAME_LEN - 1 = 254 (con largo par dividiría entre FRAME_LEN).
  const int den = (FRAME_LEN % 2 == 0) ? FRAME_LEN : FRAME_LEN - 1;
  for (int n = 0; n < FRAME_LEN; n++)
    ventana[n] = 0.5f - 0.5f * cosf(PI2 * n / den);
  for (int k = 0; k < FFT_LEN / 2; k++) {
    cos_tab[k] = cosf(PI2 * k / FFT_LEN);
    sin_tab[k] = -sinf(PI2 * k / FFT_LEN);
  }
  int bits = 0;
  while ((1 << bits) < FFT_LEN) bits++;
  for (int i = 0; i < FFT_LEN; i++) {
    int r = 0;
    for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    bit_rev[i] = r;
  }
}

// FFT compleja radix-2 en el lugar (re, im)
inline void fft() {
  for (int i = 0; i < FFT_LEN; i++) {
    int j = bit_rev[i];
    if (j > i) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int tam = 2; tam <= FFT_LEN; tam <<= 1) {
    int mitad = tam >> 1, paso = FFT_LEN / tam;
    for (int ini = 0; ini < FFT_LEN; ini += tam) {
      for (int k = 0; k < mitad; k++) {
        float wr = cos_tab[k * paso], wi = sin_tab[k * paso];
        int a = ini + k, b = a + mitad;
        float tr = re[b] * wr - im[b] * wi;
        float ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr;        im[a] += ti;
      }
    }
  }
}

inline int8_t cuantizar(float x) {
  long q = lrintf(x / ENTRADA_ESCALA) + ENTRADA_ZP;   // lrintf redondea igual que numpy (mitades al par)
  if (q < -128) q = -128;
  if (q > 127) q = 127;
  return (int8_t)q;
}

// audio: N_MUESTRAS valores int16 | salida: N_FRAMES * N_BINS valores int8 (orden frame, bin)
inline void calcular(const int16_t* audio, int8_t* salida) {
  for (int f = 0; f < N_FRAMES; f++) {
    const int16_t* trozo = audio + f * FRAME_STEP;
    for (int n = 0; n < FFT_LEN; n++) {
      re[n] = (n < FRAME_LEN) ? (trozo[n] / 32768.0f) * ventana[n] : 0.0f;
      im[n] = 0.0f;
    }
    fft();
    for (int k = 0; k < N_BINS; k++) {
      float mag = sqrtf(re[k] * re[k] + im[k] * im[k]);
      salida[f * N_BINS + k] = cuantizar((log1pf(mag) - NORM_MU) / NORM_SD);
    }
  }
}

}  // namespace espectro
