// Device-side constants shared by grxDNN's kernels.
//
// Small, but it has to be one definition rather than two: all.cpp includes
// every kernel file into a single translation unit, so a constant defined in
// two of them is a redefinition error at build time — which is how this header
// came to exist, when the mask kernel needed the same negative infinity the
// softmax kernel already had.

#ifndef GRXDNN_KERNELS_DNN_DEVICE_H
#define GRXDNN_KERNELS_DNN_DEVICE_H

namespace grxdnn_dev {

// The largest finite float, negated.
//
// It is the identity for a max reduction — a lane with no element must
// contribute something that cannot win, and 0 would win over a row of
// negatives — and it is also what a masked attention score is set to. Not an
// actual -inf: dev_exp clamps below -88 and returns exactly zero, so a masked
// entry contributes nothing to the sum, whereas a true infinity would give
// inf - inf = NaN in the max subtraction the moment a whole row was masked.
constexpr float kNegInf = -3.402823466e+38f;

}  // namespace grxdnn_dev

#endif  // GRXDNN_KERNELS_DNN_DEVICE_H
