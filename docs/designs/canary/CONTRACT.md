# The first board's canary: its numerical contract

What every engine of the first board's three-engine chain has to compute,
exactly, for the canary in this folder, and what Gate F holds the PTA to. It
is written from the reference implementation (`harness/pta_mnist.c`,
`golden_image()`) and checked against it by an implementation written apart
(GRXCP `docs/designs/pta_board_canary_golden.py`): the two agree on every byte
of every tensor here. The owner's rulings it rests on are GRXCP's B20, B21
and B22. Integer arithmetic throughout; no floating point is used for any
declared-format value.

## Files and encodings

Every file is headerless raw bytes, little-endian, row-major, with the shape
below. A vector is one of the 1,000 acceptance vectors, in file order; the
calibration set's files (`cal_`) have the same layout over its 1,000.

| File | Shape | Type | Holds |
|---|---|---|---|
| `pixels_u8.bin` | [1000, 64] | uint8 | The image pooled to 8 x 8, row by row |
| `inputs_int8.bin` | [1000, 64] | int8, 0 to 127 | FC1's inputs |
| `fc1_weights_int8.bin` | [64, 64] | int8, -128 to 127 | FC1's weights, [output, input] |
| `fc1_bias_int32.bin` | [64] | int32 | FC1's biases, in its sums' units |
| `fc1_relu_int8.bin` | [1000, 64] | int8, 0 to 127 | FC1's outputs |
| `fc2_weights_int8.bin` | [64, 64] | int8, -128 to 127 | FC2's weights, [output, input] |
| `fc2_bias_int32.bin` | [64] | int32 | FC2's biases, in its sums' units |
| `fc2_relu_int8.bin` | [1000, 64] | int8, 0 to 127 | FC2's outputs |
| `fc3_act_codes_u8.bin` | [1000, 64] | uint8, 0 to 31 | FC3's activation codes |
| `fc3_weights_int8.bin` | [16, 64] | int8, -32 to 31 | FC3's weight codes, [output, input] |
| `fc3_bias_int32.bin` | [16] | int32 | FC3's biases, in units of 2^-9 of a code |
| `fc3_sums_int32.bin` | [1000, 16] | int32 | FC3's exact sums |
| `fc3_codes_int8.bin` | [1000, 16] | int8, -128 to 127 | FC3's declared reference |
| `class_ids_u8.bin` | [1000] | uint8, 0 to 15 | The declared reference's class |
| `labels_u8.bin` | [1000] | uint8, 0 to 9 | MNIST's labels |
| `highprec_logits_f64.bin` | [1000, 16] | IEEE-754 binary64 | The network as trained, for quality only |

## The chain, step by step

All sums are exact. Each fits in an int32 with room: FC1's and FC2's are 64
products of at most 127 x 128 and a bias, FC3's 64 of at most 31 x 32.

1. **The input.** Each 28 x 28 MNIST image is cut to its middle 24 x 24 (rows
   and columns 2 to 25) and pooled in 3 x 3 blocks: a block's pixel is
   (2s + 9) div 18 of its sum s, its mean rounded half up. A pooled pixel p
   becomes the input (127p + 127) div 255, from 0 to 127. Feature r x 8 + c
   is the block in row r and column c.
2. **FC1**, on the GRX930 NPU. For each output n: v = the sum over k of
   input[k] x w1[n, k], plus b1[n]. Then the ReLU, v = max(v, 0). Then the
   rescale by 2^9, rounded half up: (v + 256) >> 9. Then the hold at 127:
   output = min(that, 127).
3. **FC2**, on the G100: the same, on FC1's outputs, with w2, b2 and the same
   shift of 9.
4. **The conversion**, on the CPU: FC2's output x, from 0 to 127, becomes
   FC3's activation code min((x + 2) >> 2, 31), from 0 to 31: the top six
   bits, rounded half up and held.
5. **FC3**, on the PTA. For each output o: s = the sum over k of code[k] x
   w3[o, k], with no bias. The declared reference is the 8-bit converter's
   code floor(s / 32 + 1/2), held to -128 to 127. A code at either end is a
   range fault; no code of the acceptance or calibration vectors is held, and
   they run from -100 to 67.
6. **The class**, on the CPU: the output o for which code[o] x 512 + b3[o]
   is largest; of two the same, the lower o. All 16 outputs take part;
   outputs 10 to 15 were trained toward no class.

## Gate F

The PTA's per-run figures are taken against FC3's declared reference over all
1,000 vectors and 16 outputs, one shot a vector and no averaging, as B19, B21
and B22 rule them:

- **The output scale.** One number for all 16 outputs, fixed once at
  calibration: the least-squares slope through the origin of the calibration
  set's reference codes on the PTA's codes for the same vectors. It is never
  fitted on the acceptance vectors.
- **The rms error**: 100 x sqrt(sum of (scale x y - r)^2 / sum of r^2), y the
  PTA's codes and r the reference's. Gate F's band: 2.5% to 7.3%, both edges
  inside. Reported also with a scale of one.
- **The class**: the class of scale x y x 512 + b3, as in step 6, against the
  reference's class, on at least 98.5% of the vectors.
- **The tail**: an output is t codes off or more when |scale x y - r| is more
  than t - 1/2. At most 16 of a run's 16,000 outputs may be 17 or more codes
  off.
- **Repeat to repeat** is reported beside these, not in them.
- **A held PTA code** is a range fault, not an error to average.

## What is not in it

- **The engines' own stage outputs.** FC1 on the NPU and FC2 on the G100
  have to give `fc1_relu_int8.bin` and `fc2_relu_int8.bin` exactly. No board
  has run them yet; as the owner ruled, they are tested against these files in
  integration, after the freeze.
- **The PTA's register settings** for this layer: its tile is 128 x 64, with
  6-bit activations and weights and the 8-bit converter at a step of 2^9 of
  its operands' sums, and how a board writes that is its own.
- **Review and approval.** Both are recorded in the manifest when made.
