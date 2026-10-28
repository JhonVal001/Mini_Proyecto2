// =====================================================================
//  PASTILLERO POR VOZ PARA PERSONAS CIEGAS  (prototipo educativo)
//  Arduino Nano 33 BLE Sense (núcleo Mbed OS) + modelo entrenado en Colab
//
//  Cableado:   servo D8 | zumbador D3 (por la resistencia al 2N2222) | pulsador NC entre D2 y GND
//              MT3608 a 5 V -> pin VIN del Nano (NO al pin 5V), servo y zumbador; GND común
//  Placa:      Boards Manager -> "Arduino Mbed OS Nano Boards" -> Arduino Nano 33 BLE
//  Librerías:  Library Manager -> "Chirale_TensorFlowLite" y "Servo"
//  Archivos que deben estar en esta misma carpeta (salen de Colab, NO editar):
//     modelo_pastillas.h, config_pastillas.h, vector_prueba.h, espectrograma.h
//  (No se usa ningún header de Edge Impulse: el modelo es el de Colab.)
//
//  Monitor serie a 115200. Comandos de prueba:
//     1 / 2 / 3 -> mueve el servo a la posición A / B / C
//     b         -> prueba el zumbador
//     g         -> escuchar 1.5 s sin usar el pulsador
//     p         -> muestra lo que lee el pulsador (0/1)
//     t         -> autoprueba del modelo con un audio de Colab
// =====================================================================
#include <PDM.h>
#include <Servo.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "config_pastillas.h"
#include "modelo_pastillas.h"
#include "vector_prueba.h"
#include "espectrograma.h"


// #####################################################################
//  CONSTANTES (edítalas a tu gusto)
// #####################################################################
// ---- Pines ----
const int PIN_SERVO    = 8;     // señal del servo
const int PIN_PULSADOR = 2;     // pulsador NORMALMENTE CERRADO entre D2 y GND (INPUT_PULLUP)
const int PIN_ZUMBADOR = 3;     // D3 -> resistencia 470 ohm -> base del 2N2222 (zumbador)

// ---- Lógica del pulsador ----
// Pulsador NC a GND con INPUT_PULLUP:
//   en reposo el contacto está cerrado -> D2 unido a GND -> se lee LOW (0)
//   al presionar el contacto se abre   -> la resistencia interna sube D2 -> se lee HIGH (1)
// Por eso "presionado" = HIGH. Si al probar notas que está al revés
// (escucha sin presionar), cambia esta constante a LOW.
// Ventaja del NC: si el cable del pulsador se suelta, D2 queda en HIGH y el
// monitor serie lo avisa al encender.
const int NIVEL_PRESIONADO = HIGH;

// ---- Servo (180°: 3 zonas de 60°, al centro de cada una) ----
const int ANGULO_A      = 30;
const int ANGULO_B      = 90;
const int ANGULO_C      = 150;
const int ANGULO_INICIO = 90;   // posición al encender
const int ANGULO_MIN    = 5;    // límites de seguridad: nunca fuera de [5°, 175°]
const int ANGULO_MAX    = 175;
const int MS_POR_GRADO  = 15;   // se mueve de 1° en 1° con esta pausa
const int MS_ESPERA_LLEGADA = 400;   // espera al llegar antes de detach()

// ---- Reconocimiento ----
const float UMBRAL = 0.85f;     // confianza mínima de la mejor clase

// ---- Zumbador ----
// USAR_TONE = false: el pitido se genera a mano (recomendado, ver nota abajo).
// USAR_TONE = true : usa tone()/noTone() del núcleo Mbed.
// NOTA: en el núcleo Mbed, noTone() no libera la memoria del pin (cada tone() deja
// unos bytes ocupados). Con miles de pitidos sin apagar el aparato, se acabaría la RAM.
const bool USAR_TONE = false;
const int FRECUENCIA_HZ = 2500;
const int MS_PITIDO     = 120;  // pitido corto = 120 ms de tono + 120 ms de pausa
const int MS_PAUSA      = 120;

// ---- Pulsador ----
const int MS_ANTIRREBOTE = 30;

// ---- Palabras y acciones ----
// MOVER       -> lleva el servo al ángulo y luego 2 pitidos
// SOLO_PITIDO -> no mueve el servo, solo 2 pitidos
enum Accion { MOVER, SOLO_PITIDO };
struct Palabra { const char* nombre; Accion accion; int angulo; };
const Palabra PALABRAS[] = {
  {"dolex",     MOVER,       ANGULO_A},   // [PALABRA_1] -> posición A
  {"aspirina",  MOVER,       ANGULO_B},   // [PALABRA_2] -> posición B
  {"omeprazol", MOVER,       ANGULO_C},   // [PALABRA_3] -> posición C
  {"advil",     SOLO_PITIDO, 0},          // [PALABRA_4] solo zumbador
  {"losartan",  SOLO_PITIDO, 0},          // [PALABRA_5] solo zumbador
};
const int N_PALABRAS = sizeof(PALABRAS) / sizeof(PALABRAS[0]);

// Grabación tipo "walkie-talkie": graba MIENTRAS el pulsador está presionado
// y termina al soltarlo. Luego el modelo analiza el segundo con más voz.
// - Si se suelta antes de 1 s, el resto se rellena con silencio.
// - Si se mantiene más de SEGUNDOS_MAX, la grabación se corta sola (límite de RAM)
//   y se clasifica cuando se suelte el pulsador.
const float SEGUNDOS_MAX = 2.5f;           // máximo que cabe en memoria (2.5 s = 80 KB)
const float SEGUNDOS_PRUEBA_SERIE = 1.5f;  // duración fija con el comando 'g' (sin pulsador)
const int   MS_MINIMO_PULSADO = 250;       // pulsaciones más cortas se ignoran (toque accidental)
const int   MS_RECORTE_FINAL = 40;         // se descarta el final: el "clic" al soltar el pulsador
const int   MS_SILENCIO_TRAS_PITIDO = 80;  // el pitido no debe quedar en la grabación


// #####################################################################
//  ZUMBADOR Y SERVO
// #####################################################################
Servo servo;
int anguloActual = ANGULO_INICIO;   // tras detach() el servo no informa su posición: la recordamos

void pitidoCorto() {
  if (USAR_TONE) {
    tone(PIN_ZUMBADOR, FRECUENCIA_HZ);
    delay(MS_PITIDO);
    noTone(PIN_ZUMBADOR);
  } else {
    // onda cuadrada a FRECUENCIA_HZ durante MS_PITIDO (bloqueante; nada más corre a la vez)
    const unsigned long medio_us = 500000UL / FRECUENCIA_HZ;
    const unsigned long ciclos = (unsigned long)MS_PITIDO * FRECUENCIA_HZ / 1000UL;
    for (unsigned long i = 0; i < ciclos; i++) {
      digitalWrite(PIN_ZUMBADOR, HIGH); delayMicroseconds(medio_us);
      digitalWrite(PIN_ZUMBADOR, LOW);  delayMicroseconds(medio_us);
    }
  }
  digitalWrite(PIN_ZUMBADOR, LOW);  // SIEMPRE apagado al final: noTone() de Mbed puede dejar el pin en alto
  delay(MS_PAUSA);
}

void pitidos(int n) { for (int i = 0; i < n; i++) pitidoCorto(); }

int limitar(int a) { return a < ANGULO_MIN ? ANGULO_MIN : (a > ANGULO_MAX ? ANGULO_MAX : a); }

// Mueve de 1° en 1°, espera, y suelta el servo (detach) para que no zumbe ni meta ruido.
void moverServo(int destino) {
  destino = limitar(destino);
  servo.attach(PIN_SERVO);
  servo.write(anguloActual);                 // arranca desde donde quedó (sin salto)
  int paso = (destino > anguloActual) ? 1 : -1;
  while (anguloActual != destino) {
    anguloActual += paso;
    servo.write(anguloActual);
    delay(MS_POR_GRADO);
  }
  delay(MS_ESPERA_LLEGADA);
  servo.detach();
  digitalWrite(PIN_SERVO, LOW);
}


// #####################################################################
//  RECONOCIMIENTO DE VOZ (NO cambiar: tiene que ser igual a Colab)
// #####################################################################
const int GANANCIA_PDM = 127;              // la misma del firmware de Edge Impulse con que se grabaron los datos
const int N_GRABACION = (int)(FS * SEGUNDOS_MAX);
static int16_t grabacion[N_GRABACION];
volatile int n_grabadas = 0;
volatile bool grabando = false;
static int16_t buffer_pdm[512];

constexpr int TAM_ARENA = 90 * 1024;
alignas(16) static uint8_t arena[TAM_ARENA];
tflite::MicroInterpreter* interprete = nullptr;
TfLiteTensor* entrada = nullptr;
TfLiteTensor* salida = nullptr;

void alRecibirPDM() {
  int bytes = PDM.available();
  if (bytes > (int)sizeof(buffer_pdm)) bytes = sizeof(buffer_pdm);
  PDM.read(buffer_pdm, bytes);
  if (!grabando) return;
  int n = bytes / 2;
  for (int i = 0; i < n && n_grabadas < N_GRABACION; i++) grabacion[n_grabadas++] = buffer_pdm[i];
  if (n_grabadas >= N_GRABACION) grabando = false;
}

void detener(const char* msg) {
  Serial.println(msg);
  while (true) { pitidos(1); delay(1500); }
}

int clasificar(const int16_t* audio, float* prob) {
  espectro::calcular(audio, entrada->data.int8);
  if (interprete->Invoke() != kTfLiteOk) detener("Error en Invoke()");
  int mejor = 0;
  for (int i = 0; i < N_CLASES; i++) {
    prob[i] = (salida->data.int8[i] - SALIDA_ZP) * SALIDA_ESCALA;
    if (prob[i] > prob[mejor]) mejor = i;
  }
  return mejor;
}

void autoprueba() {
  float prob[N_CLASES];
  int k = clasificar(PRUEBA_AUDIO, prob);
  int dif_max = 0;
  for (int i = 0; i < N_FRAMES * N_BINS; i++) dif_max = max(dif_max, abs((int)entrada->data.int8[i] - (int)PRUEBA_ENTRADA[i]));
  bool ok = (strcmp(CLASES[k], PRUEBA_CLASE) == 0) && dif_max <= 2;
  Serial.print("Autoprueba: audio '"); Serial.print(PRUEBA_CLASE); Serial.print("' -> '"); Serial.print(CLASES[k]);
  Serial.print("' ("); Serial.print(prob[k] * 100, 1); Serial.print(" %), diferencia con Colab: "); Serial.print(dif_max);
  Serial.println(ok ? "  -> OK" : "  -> REVISAR");
}

// Devuelve true si el pulsador está presionado de forma estable (antirrebote).
bool pulsadorPresionado() {
  if (digitalRead(PIN_PULSADOR) != NIVEL_PRESIONADO) return false;
  delay(MS_ANTIRREBOTE);
  return digitalRead(PIN_PULSADOR) == NIVEL_PRESIONADO;
}
bool pulsadorSuelto() {
  if (digitalRead(PIN_PULSADOR) == NIVEL_PRESIONADO) return false;
  delay(MS_ANTIRREBOTE);
  return digitalRead(PIN_PULSADOR) != NIVEL_PRESIONADO;
}

// Graba y devuelve el índice de la clase (-1 si no hubo voz, -2 si fue un toque accidental).
// conPulsador = true : graba mientras el pulsador siga presionado.
// conPulsador = false: graba SEGUNDOS_PRUEBA_SERIE (comando 'g' del monitor serie).
int escuchar(float* prob, bool conPulsador) {
  delay(MS_SILENCIO_TRAS_PITIDO);
  unsigned long t0 = millis();
  noInterrupts(); n_grabadas = 0; grabando = true; interrupts();

  if (conPulsador) {
    bool avisado = false;
    while (!pulsadorSuelto()) {                    // graba hasta que lo suelte
      if (!grabando && !avisado) {                 // se llenó la memoria: deja de grabar pero espera a que suelte
        Serial.println("Grabación al máximo: suelta el pulsador.");
        avisado = true;
      }
      delay(5);
    }
  } else {
    while (grabando && millis() - t0 < (unsigned long)(SEGUNDOS_PRUEBA_SERIE * 1000)) delay(5);
  }
  noInterrupts(); grabando = false; int n = n_grabadas; interrupts();
  unsigned long duracion = millis() - t0;

  Serial.print("Grabado: "); Serial.print(n / (float)FS, 2); Serial.println(" s");
  if (conPulsador && duracion < (unsigned long)MS_MINIMO_PULSADO) return -2;

  n -= FS * MS_RECORTE_FINAL / 1000;               // quita el clic de soltar el pulsador
  if (n < 0) n = 0;
  if (n < N_MUESTRAS) {                            // menos de 1 s: rellena con silencio
    for (int i = n; i < N_MUESTRAS; i++) grabacion[i] = 0;
    n = N_MUESTRAS;
  }

  int mejor_ini = 0; double mejor_e = -1;      // el segundo con más energía (pasos de 10 ms)
  for (int ini = 0; ini + N_MUESTRAS <= n; ini += 160) {
    double e = 0;
    for (int i = 0; i < N_MUESTRAS; i += 4) { double v = grabacion[ini + i]; e += v * v; }
    if (e > mejor_e) { mejor_e = e; mejor_ini = ini; }
  }
  const int16_t* ventana = grabacion + mejor_ini;
  double s = 0;
  for (int i = 0; i < N_MUESTRAS; i++) { float v = ventana[i] / 32768.0f; s += v * v; }
  float rms = sqrt(s / N_MUESTRAS);
  Serial.print("Nivel de voz (rms): "); Serial.print(rms, 3); Serial.println("   (entrenamiento ~0.2)");
  if (rms < RMS_MIN) return -1;
  return clasificar(ventana, prob);
}


// #####################################################################
//  FLUJO PRINCIPAL
// #####################################################################
int buscarPalabra(const char* nombre) {
  for (int i = 0; i < N_PALABRAS; i++) if (strcmp(PALABRAS[i].nombre, nombre) == 0) return i;
  return -1;
}

void esperarSoltarPulsador() {
  while (digitalRead(PIN_PULSADOR) == NIVEL_PRESIONADO) delay(10);
  delay(MS_ANTIRREBOTE);
}

void cicloDeEscucha(bool conPulsador) {
  pitidoCorto();                                   // "te escucho": habla DESPUÉS del pitido
  if (conPulsador) Serial.println("Grabando... (suelta el pulsador al terminar)");
  float prob[N_CLASES];
  int k = escuchar(prob, conPulsador);

  if (k == -2) {
    Serial.println("Pulsación muy corta | Acción: ninguna");
    return;
  }
  if (k < 0) {                                     // silencio total: lo tratamos como "ruido"
    Serial.println("Clase: (silencio) | Acción: ninguna");
    return;
  }
  for (int i = 0; i < N_CLASES; i++) {
    Serial.print("   "); Serial.print(CLASES[i]); Serial.print(": "); Serial.print(prob[i] * 100, 1); Serial.println(" %");
  }
  const char* clase = CLASES[k];
  float conf = prob[k];
  Serial.print("Clase: "); Serial.print(clase); Serial.print(" | Confianza: "); Serial.print(conf, 2);

  if (strcmp(clase, "ruido") == 0) {               // solo si el modelo tiene la clase "ruido"
    Serial.println(" | Acción: ninguna (ruido)");
    return;
  }
  int p = buscarPalabra(clase);
  if (conf < UMBRAL || strcmp(clase, "desconocido") == 0 || p < 0) {
    Serial.println(" | Acción: 3 pitidos (repite)");
    pitidos(3);
    return;
  }
  if (PALABRAS[p].accion == MOVER) {
    Serial.print(" | Acción: servo a "); Serial.print(limitar(PALABRAS[p].angulo)); Serial.println("° + 2 pitidos");
    moverServo(PALABRAS[p].angulo);                // primero el servo...
    pitidos(2);                                    // ...y después el zumbador (nunca a la vez)
  } else {
    Serial.println(" | Acción: 2 pitidos (sin mover)");
    pitidos(2);
  }
}

void setup() {
  Serial.begin(115200);
  unsigned long t = millis();
  while (!Serial && millis() - t < 3000) {}        // funciona también sin computador

  pinMode(PIN_PULSADOR, INPUT_PULLUP);
  pinMode(PIN_ZUMBADOR, OUTPUT);
  digitalWrite(PIN_ZUMBADOR, LOW);

  // 1. Servo a 90° y suelto
  servo.attach(PIN_SERVO);
  servo.write(ANGULO_INICIO);
  anguloActual = ANGULO_INICIO;
  delay(MS_ESPERA_LLEGADA + 300);
  servo.detach();

  // Modelo
  espectro::iniciar();
  const tflite::Model* modelo = tflite::GetModel(g_modelo_pastillas);
  if (modelo->version() != TFLITE_SCHEMA_VERSION) detener("El modelo y la librería tienen versiones distintas");
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interp(modelo, resolver, arena, TAM_ARENA);
  interprete = &interp;
  if (interprete->AllocateTensors() != kTfLiteOk) detener("AllocateTensors() falló: sube TAM_ARENA");
  entrada = interprete->input(0);
  salida = interprete->output(0);

  // Micrófono (siempre encendido; solo se guarda audio mientras 'grabando' es true)
  PDM.onReceive(alRecibirPDM);
  if (!PDM.begin(1, FS)) detener("No se pudo iniciar el micrófono PDM");
  PDM.setGain(GANANCIA_PDM);

  autoprueba();
  for (int i = 0; i < N_PALABRAS; i++) {           // aviso si una palabra no existe en el modelo
    bool existe = false;
    for (int c = 0; c < N_CLASES; c++) if (strcmp(CLASES[c], PALABRAS[i].nombre) == 0) existe = true;
    if (!existe) { Serial.print("AVISO: la palabra '"); Serial.print(PALABRAS[i].nombre); Serial.println("' no está en el modelo"); }
  }
  Serial.print("Pulsador en reposo lee: "); Serial.println(digitalRead(PIN_PULSADOR));
  if (digitalRead(PIN_PULSADOR) == NIVEL_PRESIONADO)
    Serial.println("AVISO: D2 se lee como 'presionado' sin tocarlo. Revisa el cable del pulsador o cambia NIVEL_PRESIONADO.");
  Serial.println("Listo. Presiona el pulsador y di la pastilla.");
  pitidoCorto();                                   // "listo"
}

void loop() {
  // 2. Reposo: espera el pulsador (con antirrebote)
  // Al presionar: pitido y graba mientras siga presionado; al soltar, clasifica.
  if (pulsadorPresionado()) {
    cicloDeEscucha(true);
    esperarSoltarPulsador();                       // si lo presionan mientras suena o se mueve el servo, se ignora
  }

  // Comandos de prueba por el monitor serie
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1')      { Serial.println("Prueba: posición A"); moverServo(ANGULO_A); }
    else if (c == '2') { Serial.println("Prueba: posición B"); moverServo(ANGULO_B); }
    else if (c == '3') { Serial.println("Prueba: posición C"); moverServo(ANGULO_C); }
    else if (c == 'b' || c == 'B') { Serial.println("Prueba: zumbador"); pitidos(2); }
    else if (c == 'g' || c == 'G') cicloDeEscucha(false);
    else if (c == 't' || c == 'T') autoprueba();
    else if (c == 'p' || c == 'P') { Serial.print("Pulsador D2 = "); Serial.println(digitalRead(PIN_PULSADOR)); }
  }
}
