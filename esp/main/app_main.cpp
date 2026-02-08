#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_timer.h"
#include "esp_log.h"

static const char *TAG = "FIRE_SNN";

// Model config
static constexpr uint32_t TIMESTEPS = 35;               // training uses num_steps = 35
static constexpr int32_t MARGIN_FOR_EARLY_DECISION = 3; // spike margin for early decision estimate

// Embedded binaries (from CMake EMBED_FILES)
extern const uint8_t snn_weights_bin_start[] asm("_binary_snn_weights_bin_start");
extern const uint8_t snn_weights_bin_end[] asm("_binary_snn_weights_bin_end");

extern const uint8_t replay_data_bin_start[] asm("_binary_replay_data_bin_start");
extern const uint8_t replay_data_bin_end[] asm("_binary_replay_data_bin_end");

// Tiny RNG for Bernoulli spike generation (deterministic per sample)
struct XorShift32
{
    uint32_t state;
    inline uint32_t next_u32()
    {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
    inline float next_f01()
    {
        // 24-bit mantissa uniform [0,1)
        return (float)((next_u32() >> 8) & 0x00FFFFFF) * (1.0f / 16777216.0f);
    }
};

static inline float bernoulli_spike(float p, XorShift32 *rng)
{
    if (p <= 0.0f)
        return 0.0f;
    if (p >= 1.0f)
        return 1.0f;
    return (rng->next_f01() < p) ? 1.0f : 0.0f;
}

// Packed binary formats
#pragma pack(push, 1)
struct WeightsHeader
{
    char magic[4]; // "SNNW"
    uint32_t in_dim;
    uint32_t h1;
    uint32_t h2;
    uint32_t out_dim;
    float beta;
    float thr;
    // Followed by float32 blobs in order:
    // fc1_w (h1*in_dim), fc1_b (h1),
    // fc2_w (h2*h1),    fc2_b (h2),
    // fco_w (out_dim*h2), fco_b (out_dim)
};

struct DataHeader
{
    char magic[4]; // "DATA"
    uint32_t n_samples;
    uint32_t n_features; // should be 4
    // Followed by float32 features [n_samples*n_features],
    // then uint8 labels [n_samples]
};
#pragma pack(pop)

struct ModelWeights
{
    uint32_t in_dim, h1, h2, out_dim;
    float beta, thr;
    const float *fc1_w;
    const float *fc1_b;
    const float *fc2_w;
    const float *fc2_b;
    const float *fco_w;
    const float *fco_b;
};

struct ReplayData
{
    uint32_t n_samples;
    uint32_t n_features;
    const float *x;   // [n_samples*n_features]
    const uint8_t *y; // [n_samples]
};

static bool load_weights(ModelWeights *mw)
{
    const uint8_t *start = snn_weights_bin_start;
    const uint8_t *end = snn_weights_bin_end;
    size_t len = (size_t)(end - start);

    if (len < sizeof(WeightsHeader))
    {
        ESP_LOGE(TAG, "Weights blob too small: %u bytes", (unsigned)len);
        return false;
    }

    WeightsHeader hdr;
    memcpy(&hdr, start, sizeof(hdr));
    if (memcmp(hdr.magic, "SNNW", 4) != 0)
    {
        ESP_LOGE(TAG, "Bad weights magic");
        return false;
    }

    mw->in_dim = hdr.in_dim;
    mw->h1 = hdr.h1;
    mw->h2 = hdr.h2;
    mw->out_dim = hdr.out_dim;
    mw->beta = hdr.beta;
    mw->thr = hdr.thr;

    const uint8_t *p = start + sizeof(WeightsHeader);

    auto need_bytes = [&](size_t nbytes) -> bool
    {
        return (p + nbytes) <= end;
    };

    size_t fc1_w_n = (size_t)mw->h1 * (size_t)mw->in_dim;
    size_t fc1_b_n = (size_t)mw->h1;
    size_t fc2_w_n = (size_t)mw->h2 * (size_t)mw->h1;
    size_t fc2_b_n = (size_t)mw->h2;
    size_t fco_w_n = (size_t)mw->out_dim * (size_t)mw->h2;
    size_t fco_b_n = (size_t)mw->out_dim;

    size_t total_floats = fc1_w_n + fc1_b_n + fc2_w_n + fc2_b_n + fco_w_n + fco_b_n;
    size_t total_bytes = total_floats * sizeof(float);

    if (!need_bytes(total_bytes))
    {
        ESP_LOGE(TAG, "Weights blob truncated. Need %u bytes after header, have %u",
                 (unsigned)total_bytes, (unsigned)(end - p));
        return false;
    }

    mw->fc1_w = (const float *)p;
    p += fc1_w_n * sizeof(float);
    mw->fc1_b = (const float *)p;
    p += fc1_b_n * sizeof(float);
    mw->fc2_w = (const float *)p;
    p += fc2_w_n * sizeof(float);
    mw->fc2_b = (const float *)p;
    p += fc2_b_n * sizeof(float);
    mw->fco_w = (const float *)p;
    p += fco_w_n * sizeof(float);
    mw->fco_b = (const float *)p;
    p += fco_b_n * sizeof(float);

    ESP_LOGI(TAG, "Loaded weights: in=%u h1=%u h2=%u out=%u beta=%.3f thr=%.3f",
             (unsigned)mw->in_dim, (unsigned)mw->h1, (unsigned)mw->h2, (unsigned)mw->out_dim,
             (double)mw->beta, (double)mw->thr);
    return true;
}

static bool load_replay(ReplayData *rd)
{
    const uint8_t *start = replay_data_bin_start;
    const uint8_t *end = replay_data_bin_end;
    size_t len = (size_t)(end - start);

    if (len < sizeof(DataHeader))
    {
        ESP_LOGE(TAG, "Replay blob too small: %u bytes", (unsigned)len);
        return false;
    }

    DataHeader hdr;
    memcpy(&hdr, start, sizeof(hdr));
    if (memcmp(hdr.magic, "DATA", 4) != 0)
    {
        ESP_LOGE(TAG, "Bad replay magic");
        return false;
    }

    rd->n_samples = hdr.n_samples;
    rd->n_features = hdr.n_features;

    const uint8_t *p = start + sizeof(DataHeader);

    size_t feat_floats = (size_t)rd->n_samples * (size_t)rd->n_features;
    size_t feat_bytes = feat_floats * sizeof(float);
    size_t label_bytes = (size_t)rd->n_samples * sizeof(uint8_t);

    if ((p + feat_bytes + label_bytes) > end)
    {
        ESP_LOGE(TAG, "Replay blob truncated");
        return false;
    }

    rd->x = (const float *)p;
    p += feat_bytes;
    rd->y = (const uint8_t *)p;

    ESP_LOGI(TAG, "Loaded replay: n=%u features=%u", (unsigned)rd->n_samples, (unsigned)rd->n_features);
    return true;
}

// Dense layer (row-major W[out][in])
static inline void dense(const float *W, const float *b,
                         const float *x, float *y,
                         uint32_t out_dim, uint32_t in_dim)
{
    for (uint32_t i = 0; i < out_dim; i++)
    {
        float acc = b ? b[i] : 0.0f;
        const float *Wi = W + (size_t)i * (size_t)in_dim;
        for (uint32_t j = 0; j < in_dim; j++)
            acc += Wi[j] * x[j];
        y[i] = acc;
    }
}

// LIF step (subtract-threshold reset)
static inline float lif_step(float cur, float *mem, float beta, float thr)
{
    float m = beta * (*mem) + cur;
    float spk = (m > thr) ? 1.0f : 0.0f;
    if (spk > 0.0f)
        m -= thr;
    *mem = m;
    return spk;
}

// --- Static buffers (keep off stack) ---
static float mem1[500];
static float mem2[500];
static float mem3[2];

static float in_spk[4];
static float cur1[500];
static float spk1[500];
static float cur2[500];
static float spk2[500];
static float cur3[2];
static float spk3[2];

static void infer_task(void *arg)
{
    ModelWeights mw;
    ReplayData rd;

    if (!load_weights(&mw) || !load_replay(&rd))
    {
        ESP_LOGE(TAG, "Failed to load embedded model/data.");
        vTaskDelete(NULL);
        return;
    }
    if (rd.n_features != mw.in_dim)
    {
        ESP_LOGE(TAG, "Replay features (%u) != model input dim (%u)",
                 (unsigned)rd.n_features, (unsigned)mw.in_dim);
        vTaskDelete(NULL);
        return;
    }

    // If you ever embed a different model, these arrays are sized for 4-500-500-2.
    if (mw.in_dim != 4 || mw.h1 != 500 || mw.h2 != 500 || mw.out_dim != 2)
    {
        ESP_LOGE(TAG, "Expected model dims 4-500-500-2, got %u-%u-%u-%u",
                 (unsigned)mw.in_dim, (unsigned)mw.h1, (unsigned)mw.h2, (unsigned)mw.out_dim);
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "Starting CONTINUOUS replay streaming...");
    ESP_LOGI(TAG, "Tip: quit monitor with Ctrl+]  (or Ctrl+T then Ctrl+H for help)");

    uint32_t idx = 0;

    // Running stats (optional)
    uint32_t TP = 0, TN = 0, FP = 0, FN = 0;
    uint32_t printed = 0;
    double total_ms = 0.0;

    while (true)
    {
        // Reset neuron states
        memset(mem1, 0, sizeof(mem1));
        memset(mem2, 0, sizeof(mem2));
        memset(mem3, 0, sizeof(mem3));

        int out_count[2] = {0, 0};
        int diff_by_step[TIMESTEPS] = {0};

        // Deterministic RNG per sample index
        XorShift32 rng;
        rng.state = 0xC0FFEEu ^ (idx * 2654435761u);

        const float *x = rd.x + (size_t)idx * (size_t)rd.n_features;
        uint8_t y = rd.y[idx];

        int64_t t0 = esp_timer_get_time();

        for (uint32_t t = 0; t < TIMESTEPS; t++)
        {
            // Bernoulli spikes from probabilities
            for (uint32_t j = 0; j < mw.in_dim; j++)
            {
                in_spk[j] = bernoulli_spike(x[j], &rng);
            }

            dense(mw.fc1_w, mw.fc1_b, in_spk, cur1, mw.h1, mw.in_dim);
            for (uint32_t i = 0; i < mw.h1; i++)
                spk1[i] = lif_step(cur1[i], &mem1[i], mw.beta, mw.thr);

            dense(mw.fc2_w, mw.fc2_b, spk1, cur2, mw.h2, mw.h1);
            for (uint32_t i = 0; i < mw.h2; i++)
                spk2[i] = lif_step(cur2[i], &mem2[i], mw.beta, mw.thr);

            dense(mw.fco_w, mw.fco_b, spk2, cur3, mw.out_dim, mw.h2);
            for (uint32_t k = 0; k < mw.out_dim; k++)
            {
                spk3[k] = lif_step(cur3[k], &mem3[k], mw.beta, mw.thr);
                out_count[k] += (int)spk3[k];
            }

            diff_by_step[t] = out_count[1] - out_count[0];
        }

        int64_t t1 = esp_timer_get_time();
        double ms = (double)(t1 - t0) / 1000.0;
        total_ms += ms;

        int pred = (out_count[1] > out_count[0]) ? 1 : 0;

        // Decision step estimate
        int final_sign = (pred == 1) ? 1 : -1;
        int decision_step = -1;
        for (uint32_t t = 0; t < TIMESTEPS; t++)
        {
            int d = diff_by_step[t];
            if ((d * final_sign) >= MARGIN_FOR_EARLY_DECISION)
            {
                decision_step = (int)t;
                break;
            }
        }

        // Update confusion matrix
        if (y == 1 && pred == 1)
            TP++;
        else if (y == 0 && pred == 0)
            TN++;
        else if (y == 0 && pred == 1)
            FP++;
        else if (y == 1 && pred == 0)
            FN++;

        // Print every sample for now (you can change this to every N if spammy)
        int diff = out_count[1] - out_count[0];
        ESP_LOGI(TAG, "idx=%u y=%u pred=%d diff=%d spikes=[%d,%d] time=%.2fms decision_step=%d",
                 (unsigned)idx, (unsigned)y, pred, diff, out_count[0], out_count[1], ms, decision_step);

        printed++;
        if (printed % 20 == 0)
        {
            uint32_t total = TP + TN + FP + FN;
            double acc = (total > 0) ? (double)(TP + TN) / (double)total : 0.0;
            double avg = (printed > 0) ? total_ms / (double)printed : 0.0;
            ESP_LOGI(TAG, "RUNNING: n=%u acc=%.3f TP=%u TN=%u FP=%u FN=%u avg_ms=%.2f",
                     (unsigned)total, acc, (unsigned)TP, (unsigned)TN, (unsigned)FP, (unsigned)FN, avg);
        }

        // next sample
        idx++;
        if (idx >= rd.n_samples)
            idx = 0;

        // yield so idle tasks run (prevents task watchdog triggers)
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

extern "C" void app_main(void)
{
    // Put inference in its own task so app_main can return cleanly.
    // Stack can be modest since large buffers are static.
    xTaskCreate(
        infer_task,
        "infer_task",
        8192,
        NULL,
        5,
        NULL);
}
