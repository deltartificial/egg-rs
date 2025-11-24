use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::arch::aarch64::*;

const VOCAB_SIZE: usize = 256;
const HIDDEN_DIM: usize = 128;
const N_LAYERS: usize = 2;
const SEQ_LEN: usize = 512;
const POPULATION_SIZE: usize = 32;
const FIXED_POINT: i32 = 4;
const SIGMA_SHIFT: i32 = 4;
const UPDATE_THRESHOLD: i32 = 40;
const MAX_VAL: i8 = 127;
const MIN_VAL: i8 = -127;

const MAX_MATRIX_DIM: usize = HIDDEN_DIM * 4;
const MAX_POP_PAIRS: usize = POPULATION_SIZE / 2;

static mut EXP2_TABLE: [i32; 256] = [0; 256];

struct Dataset {
    data: Vec<u8>,
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct RecurrentState {
    h: [[i8; HIDDEN_DIM]; N_LAYERS],
}

impl Default for RecurrentState {
    fn default() -> Self {
        Self { h: [[0; HIDDEN_DIM]; N_LAYERS] }
    }
}

#[repr(C, align(16))]
struct EggModel {
    embedding: Vec<i8>,
    gru_weights: Vec<Vec<Vec<i8>>>,
    gru_biases: Vec<Vec<Vec<i8>>>,
    mlp_weights: Vec<Vec<Vec<i8>>>,
    head: Vec<i8>,
    ln_weights: Vec<Vec<Vec<i8>>>,
    ln_out: Vec<i8>,
}

unsafe impl Send for EggModel {}
unsafe impl Sync for EggModel {}

fn init_tables() {
    unsafe {
        for i in 0..256 {
            let exp_val = 2.0_f64.powf(i as f64 / (1 << FIXED_POINT) as f64);
            EXP2_TABLE[i] = (exp_val * (1 << FIXED_POINT) as f64) as i32;
        }
    }
}

#[inline(always)]
fn clipped_add(a: i32, b: i32) -> i8 {
    (a + b).clamp(MIN_VAL as i32, MAX_VAL as i32) as i8
}

#[inline(always)]
fn xorshift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

#[inline(always)]
fn gen_noise_vector_neon(rng: &mut u32, out: &mut [i8]) {
    let len = out.len();
    let mut i = 0;

    while i + 16 <= len {
        unsafe {
            let mut vals = [0i8; 16];
            for j in 0..16 {
                let r = xorshift32(rng);
                let sign = if r & 1 != 0 { 1 } else { -1 };
                vals[j] = sign * ((r >> 1) & 31) as i8;
            }

            let vec = vld1q_s8(vals.as_ptr());
            vst1q_s8(out.as_mut_ptr().add(i), vec);
        }
        i += 16;
    }

    while i < len {
        let r = xorshift32(rng);
        let sign = if r & 1 != 0 { 1 } else { -1 };
        out[i] = sign * ((r >> 1) & 31) as i8;
        i += 1;
    }
}

#[inline(always)]
unsafe fn dot_product_neon(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len().min(b.len());
    let mut acc = vdupq_n_s32(0);
    let mut i = 0;

    while i + 16 <= len {
        let a_vec = vld1q_s8(a.as_ptr().add(i));
        let b_vec = vld1q_s8(b.as_ptr().add(i));
        let mul_low = vmull_s8(vget_low_s8(a_vec), vget_low_s8(b_vec));
        let mul_high = vmull_s8(vget_high_s8(a_vec), vget_high_s8(b_vec));
        acc = vpadalq_s16(acc, mul_low);
        acc = vpadalq_s16(acc, mul_high);
        i += 16;
    }

    let mut result = vaddvq_s32(acc);
    while i < len {
        result += a[i] as i32 * b[i] as i32;
        i += 1;
    }
    result
}

#[inline(always)]
fn matmul_perturbed(input: &[i8], weights: &[i8], output: &mut [i8], rows: usize, cols: usize, layer_seed: u32, noise_sign: i32, shift: i32, a_vec: &mut [i8], b_vec: &mut [i8]) {
    let mut rng = layer_seed;
    gen_noise_vector_neon(&mut rng, &mut a_vec[..rows]);
    gen_noise_vector_neon(&mut rng, &mut b_vec[..cols]);

    let xb = unsafe { dot_product_neon(&input[..cols.min(input.len())], &b_vec[..cols]) };

    for r in 0..rows {
        let w_row = &weights[r * cols..(r + 1) * cols];
        let mut acc = unsafe { dot_product_neon(&input[..cols.min(input.len())], w_row) };

        if noise_sign != 0 {
            let noise = (xb * a_vec[r] as i32) * noise_sign;
            acc += noise >> (FIXED_POINT + SIGMA_SHIFT);
        }

        output[r] = (acc >> shift).clamp(MIN_VAL as i32, MAX_VAL as i32) as i8;
    }
}

#[inline(always)]
fn egg_ln(x: &[i8], w: &[i8], out: &mut [i8]) {
    unsafe {
        let mut sum_v = vdupq_n_s32(0);
        let mut i = 0;

        while i + 16 <= HIDDEN_DIM {
            let xv = vld1q_s8(x.as_ptr().add(i));
            let abs_xv = vabsq_s8(xv);
            let s1 = vpaddlq_u8(vreinterpretq_u8_s8(abs_xv));
            let s2 = vpaddlq_u16(s1);
            sum_v = vaddq_s32(sum_v, vreinterpretq_s32_u32(s2));
            i += 16;
        }

        let mut sum = vaddvq_s32(sum_v);
        while i < HIDDEN_DIM {
            sum += x[i].abs() as i32;
            i += 1;
        }

        let mean = (sum / HIDDEN_DIM as i32).max(1);
        for i in 0..HIDDEN_DIM {
            out[i] = ((x[i] as i32 * w[i] as i32) / mean).clamp(MIN_VAL as i32, MAX_VAL as i32) as i8;
        }
    }
}

fn forward_pass(model: &EggModel, inputs: &[u8], targets: Option<&[u8]>, seq_len: usize, logits_out: &mut [i8], step_seed: u32, noise_sign: i32, rnn_state: &mut RecurrentState) -> i32 {
    let mut x = vec![0i8; HIDDEN_DIM];
    let mut residual = vec![0i8; HIDDEN_DIM];
    let mut buf1 = vec![0i8; HIDDEN_DIM * 4];
    let mut buf2 = vec![0i8; HIDDEN_DIM];
    let mut x_temp = vec![0i8; HIDDEN_DIM];
    let mut ft = vec![0i8; HIDDEN_DIM];
    let mut gated_past = vec![0i8; HIDDEN_DIM];
    let mut ht = vec![0i8; HIDDEN_DIM];
    let mut a_noise = vec![0i8; HIDDEN_DIM * 4];
    let mut b_noise = vec![0i8; HIDDEN_DIM * 4];
    let mut accumulated_loss = 0i32;

    for t in 0..seq_len {
        let emb_start = inputs[t] as usize * HIDDEN_DIM;
        x[..HIDDEN_DIM].copy_from_slice(&model.embedding[emb_start..emb_start + HIDDEN_DIM]);

        for l in 0..N_LAYERS {
            let l_seed = step_seed + (l as u32 * 100);

            residual[..HIDDEN_DIM].copy_from_slice(&x[..HIDDEN_DIM]);
            egg_ln(&x, &model.ln_weights[l][0], &mut x_temp);
            x[..HIDDEN_DIM].copy_from_slice(&x_temp[..HIDDEN_DIM]);

            matmul_perturbed(&x, &model.gru_weights[l][0], &mut buf1[..HIDDEN_DIM], HIDDEN_DIM, HIDDEN_DIM, l_seed + 1, noise_sign, 8, &mut a_noise, &mut b_noise);
            matmul_perturbed(&rnn_state.h[l], &model.gru_weights[l][1], &mut buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed + 2, noise_sign, 8, &mut a_noise, &mut b_noise);

            for i in 0..HIDDEN_DIM {
                ft[i] = clipped_add(clipped_add(buf1[i] as i32, buf2[i] as i32) as i32, model.gru_biases[l][0][i] as i32);
            }

            for i in 0..HIDDEN_DIM {
                gated_past[i] = (((ft[i] as i32 + 127) * rnn_state.h[l][i] as i32) >> 8) as i8;
            }

            matmul_perturbed(&x, &model.gru_weights[l][2], &mut buf1[..HIDDEN_DIM], HIDDEN_DIM, HIDDEN_DIM, l_seed + 3, noise_sign, 8, &mut a_noise, &mut b_noise);
            matmul_perturbed(&gated_past, &model.gru_weights[l][3], &mut buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed + 4, noise_sign, 8, &mut a_noise, &mut b_noise);

            for i in 0..HIDDEN_DIM {
                ht[i] = clipped_add(clipped_add(buf1[i] as i32, buf2[i] as i32) as i32, model.gru_biases[l][1][i] as i32);
            }

            for i in 0..HIDDEN_DIM {
                let update = ((ft[i] as i32 + 127) * (ht[i] as i32 - rnn_state.h[l][i] as i32)) >> 8;
                rnn_state.h[l][i] = clipped_add(rnn_state.h[l][i] as i32, update);
                x[i] = rnn_state.h[l][i];
            }

            for i in 0..HIDDEN_DIM {
                x[i] = clipped_add(x[i] as i32, residual[i] as i32);
            }

            residual[..HIDDEN_DIM].copy_from_slice(&x[..HIDDEN_DIM]);
            egg_ln(&x, &model.ln_weights[l][1], &mut x_temp);
            x[..HIDDEN_DIM].copy_from_slice(&x_temp[..HIDDEN_DIM]);

            matmul_perturbed(&x, &model.mlp_weights[l][0], &mut buf1, HIDDEN_DIM * 4, HIDDEN_DIM, l_seed + 5, noise_sign, 8, &mut a_noise, &mut b_noise);
            matmul_perturbed(&buf1, &model.mlp_weights[l][1], &mut x, HIDDEN_DIM, HIDDEN_DIM * 4, l_seed + 6, noise_sign, 9, &mut a_noise, &mut b_noise);

            for i in 0..HIDDEN_DIM {
                x[i] = clipped_add(x[i] as i32, residual[i] as i32);
            }
        }

        egg_ln(&x, &model.ln_out, &mut x_temp);
        x[..HIDDEN_DIM].copy_from_slice(&x_temp[..HIDDEN_DIM]);
        matmul_perturbed(&x, &model.head, logits_out, VOCAB_SIZE, HIDDEN_DIM, step_seed + 999, noise_sign, 8, &mut a_noise, &mut b_noise);

        if let Some(tgt) = targets {
            accumulated_loss += compute_loss(logits_out, tgt[t]);
        }
    }
    accumulated_loss
}

#[inline(always)]
fn get_msb(mut n: u32) -> i32 {
    let mut pos = 0;
    if n >= 1 << 16 { n >>= 16; pos += 16; }
    if n >= 1 << 8  { n >>= 8;  pos += 8; }
    if n >= 1 << 4  { n >>= 4;  pos += 4; }
    if n >= 1 << 2  { n >>= 2;  pos += 2; }
    if n >= 1 << 1  { pos += 1; }
    pos
}

#[inline(always)]
fn log2_fixed(x: i32) -> i32 {
    if x <= 0 { return 0; }
    let k = get_msb(x as u32);
    let fraction = if k >= 4 { (x - (1 << k)) >> (k - 4) } else { (x - (1 << k)) << (4 - k) };
    (k << 4) + fraction - 64
}

#[inline(always)]
fn compute_loss(logits: &[i8], target: u8) -> i32 {
    let mut sum_exp = 0i32;
    unsafe {
        for i in 0..VOCAB_SIZE {
            let idx = (logits[i] as i32 + 128).clamp(0, 255) as usize;
            sum_exp += EXP2_TABLE[idx];
        }
    }
    log2_fixed(sum_exp) - (logits[target as usize] as i32 + 128)
}

fn update_matrix(weights: &mut [i8], rows: usize, cols: usize, seed: u32, fitnesses: &[i32], pop_size: usize) {
    let pairs = (pop_size / 2).min(MAX_POP_PAIRS);
    let mut a_t = vec![vec![0i8; MAX_POP_PAIRS]; rows.min(MAX_MATRIX_DIM)];
    let mut b_t = vec![vec![0i8; MAX_POP_PAIRS]; cols.min(MAX_MATRIX_DIM)];

    for p in 0..pairs {
        let f = fitnesses[p];
        let mut rng = seed + p as u32;
        let mut a_temp = vec![0i8; rows];
        let mut b_temp = vec![0i8; cols];
        gen_noise_vector_neon(&mut rng, &mut a_temp);
        gen_noise_vector_neon(&mut rng, &mut b_temp);

        if f != 0 {
            for r in 0..rows { a_t[r][p] = (a_temp[r] as i32 * f) as i8; }
            for c in 0..cols { b_t[c][p] = b_temp[c]; }
        }
    }

    for r in 0..rows {
        let w_row = &mut weights[r * cols..(r + 1) * cols];
        let a_ptr = &a_t[r][..pairs];
        for c in 0..cols {
            let vote = unsafe { dot_product_neon(a_ptr, &b_t[c][..pairs]) };
            if vote > UPDATE_THRESHOLD && w_row[c] < MAX_VAL { w_row[c] += 1; }
            else if vote < -UPDATE_THRESHOLD && w_row[c] > MIN_VAL { w_row[c] -= 1; }
        }
    }
}

fn sample_logits(logits: &[i8], rng: &mut u32) -> usize {
    let mut probs = vec![0i32; VOCAB_SIZE];
    let mut sum = 0i32;
    unsafe {
        for i in 0..VOCAB_SIZE {
            let idx = (logits[i] as i32 + 128).clamp(0, 255) as usize;
            probs[i] = EXP2_TABLE[idx];
            sum += probs[i];
        }
    }
    if sum == 0 { return 0; }
    let r = (xorshift32(rng) as i32).abs() % sum;
    let mut acc = 0i32;
    for i in 0..VOCAB_SIZE {
        acc += probs[i];
        if r < acc { return i; }
    }
    VOCAB_SIZE - 1
}

fn sample_model(model: &EggModel, seed_text: &[u8], seed_len: usize, gen_len: usize) {
    let mut logits = vec![0i8; VOCAB_SIZE];
    let mut state = RecurrentState::default();
    let mut rng = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as u32;
    print!("\x1b[32m");
    let mut input_token = 0u8;
    let actual_seed_len = seed_len.min(seed_text.len());

    for t in 0..(actual_seed_len + gen_len) {
        if t < actual_seed_len {
            input_token = seed_text[t];
            if (32..=126).contains(&input_token) { print!("{}", input_token as char); }
            else { print!("."); }
        } else {
            if t == actual_seed_len { print!("\x1b[36m"); }
            if (32..=126).contains(&input_token) { print!("{}", input_token as char); }
            else { print!("."); }
        }
        forward_pass(model, &[input_token], None, 1, &mut logits, 0, 0, &mut state);
        if t >= actual_seed_len.saturating_sub(1) { input_token = sample_logits(&logits, &mut rng) as u8; }
    }
    println!("\x1b[0m");
}

fn load_data(filename: &str) -> Dataset {
    let mut file = File::open(filename).expect("Error: Create 'input.txt' first.");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read file");
    Dataset { data }
}

fn init_model() -> EggModel {
    let mut rng = 42u32;
    let mut embedding = vec![0i8; VOCAB_SIZE * HIDDEN_DIM];
    let mut head = vec![0i8; HIDDEN_DIM * VOCAB_SIZE];
    gen_noise_vector_neon(&mut rng, &mut embedding);
    gen_noise_vector_neon(&mut rng, &mut head);

    let mut gru_weights = vec![vec![vec![0i8; HIDDEN_DIM * HIDDEN_DIM]; 4]; N_LAYERS];
    let gru_biases = vec![vec![vec![0i8; HIDDEN_DIM]; 2]; N_LAYERS];
    let mut mlp_weights = vec![vec![vec![0i8; HIDDEN_DIM * (HIDDEN_DIM * 4)]; 2]; N_LAYERS];

    for l in 0..N_LAYERS {
        for g in 0..4 {
            gen_noise_vector_neon(&mut rng, &mut gru_weights[l][g]);
        }
        for m in 0..2 {
            gen_noise_vector_neon(&mut rng, &mut mlp_weights[l][m]);
        }
    }

    let ln_weights = vec![vec![vec![16i8; HIDDEN_DIM]; 2]; N_LAYERS];
    let ln_out = vec![16i8; HIDDEN_DIM];

    EggModel { embedding, gru_weights, gru_biases, mlp_weights, head, ln_weights, ln_out }
}

fn main() {
    init_tables();
    let ds = load_data("input.txt");
    println!("Loaded dataset: {} bytes", ds.data.len());

    let mut model = init_model();
    println!("Starting EGGROLL Training (Quick Bench Mode - Fast Config)...");

    let start_time = std::time::Instant::now();
    let mut total_tokens = 0usize;
    let max_steps = (ds.data.len() - 1) / SEQ_LEN;
    let mut pair_fitnesses = vec![0i32; POPULATION_SIZE / 2];
    let mut logits = vec![0i8; VOCAB_SIZE];
    let mut main_state = RecurrentState::default();

    for step in 0..max_steps {
        let step_seed = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as u32) ^ ((step as u32).wrapping_mul(0x9e3779b9));
        let start_idx = step * SEQ_LEN;
        if start_idx + SEQ_LEN + 1 >= ds.data.len() { break; }

        if step % 10 == 0 {
            if start_idx + 30 < ds.data.len() { sample_model(&model, &ds.data[start_idx..], 30, 30); }
            let loss_val = forward_pass(&model, &ds.data[start_idx..start_idx + SEQ_LEN], Some(&ds.data[start_idx + 1..start_idx + SEQ_LEN + 1]), SEQ_LEN, &mut logits, step_seed, 0, &mut main_state);
            let elapsed = start_time.elapsed().as_secs_f64();
            let tps = if elapsed > 0.0 { total_tokens as f64 / elapsed } else { 0.0 };
            println!("Step {}/{} | Loss: {:.4} | Tok/s: {:.2}", step, max_steps, loss_val as f64 / (SEQ_LEN as f64 * (1 << FIXED_POINT) as f64), tps);
        }

        let fitnesses: Vec<i32> = (0..POPULATION_SIZE/2).into_par_iter().map(|p_idx| {
            let p_seed = step_seed + p_idx as u32;
            let mut local_logits = vec![0i8; VOCAB_SIZE];
            let stride = ds.data.len() / (POPULATION_SIZE / 2);
            let mut stream_idx = (start_idx + (p_idx * stride)) % (ds.data.len() - SEQ_LEN - 1);
            if stream_idx + SEQ_LEN + 1 >= ds.data.len() { stream_idx = ds.data.len() - SEQ_LEN - 2; }

            let mut state_pos = RecurrentState::default();
            let mut state_neg = RecurrentState::default();
            let loss_pos = forward_pass(&model, &ds.data[stream_idx..stream_idx + SEQ_LEN], Some(&ds.data[stream_idx + 1..stream_idx + SEQ_LEN + 1]), SEQ_LEN, &mut local_logits, p_seed, 1, &mut state_pos);
            let loss_neg = forward_pass(&model, &ds.data[stream_idx..stream_idx + SEQ_LEN], Some(&ds.data[stream_idx + 1..stream_idx + SEQ_LEN + 1]), SEQ_LEN, &mut local_logits, p_seed, -1, &mut state_neg);

            if loss_pos < loss_neg { 1 } else if loss_neg < loss_pos { -1 } else { 0 }
        }).collect();

        pair_fitnesses.copy_from_slice(&fitnesses);

        for l in 0..N_LAYERS {
            let l_seed = step_seed.wrapping_add((l as u32).wrapping_mul(100));
            update_matrix(&mut model.gru_weights[l][0], HIDDEN_DIM, HIDDEN_DIM, l_seed.wrapping_add(1), &pair_fitnesses, POPULATION_SIZE);
            update_matrix(&mut model.gru_weights[l][1], HIDDEN_DIM, HIDDEN_DIM, l_seed.wrapping_add(2), &pair_fitnesses, POPULATION_SIZE);
            update_matrix(&mut model.gru_weights[l][2], HIDDEN_DIM, HIDDEN_DIM, l_seed.wrapping_add(3), &pair_fitnesses, POPULATION_SIZE);
            update_matrix(&mut model.gru_weights[l][3], HIDDEN_DIM, HIDDEN_DIM, l_seed.wrapping_add(4), &pair_fitnesses, POPULATION_SIZE);
            update_matrix(&mut model.mlp_weights[l][0], HIDDEN_DIM * 4, HIDDEN_DIM, l_seed.wrapping_add(5), &pair_fitnesses, POPULATION_SIZE);
            update_matrix(&mut model.mlp_weights[l][1], HIDDEN_DIM, HIDDEN_DIM * 4, l_seed.wrapping_add(6), &pair_fitnesses, POPULATION_SIZE);
        }
        update_matrix(&mut model.head, VOCAB_SIZE, HIDDEN_DIM, step_seed.wrapping_add(999), &pair_fitnesses, POPULATION_SIZE);
        total_tokens += SEQ_LEN;
    }
    println!("Training Done.");
}
