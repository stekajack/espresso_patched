// kernel generated with pystencils v0.3.3+39.g587a822, lbmpy
// v0.3.3+33.g036fe13, lbmpy_walberla/pystencils_walberla from commit ref:
// refs/heads/LeesEdwards

//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \\file CollideSweepThermalized.cpp
//! \\ingroup lbm
//! \\author lbmpy
//======================================================================================================================

#include <cmath>

#include "CollideSweepThermalized.h"
#include "core/DataTypes.h"
#include "core/Macros.h"

#include "philox_rand.h"

#define FUNC_PREFIX

#if (defined WALBERLA_CXX_COMPILER_IS_GNU) ||                                  \
    (defined WALBERLA_CXX_COMPILER_IS_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#if (defined WALBERLA_CXX_COMPILER_IS_INTEL)
#pragma warning push
#pragma warning(disable : 1599)
#endif

using namespace std;

namespace walberla {
namespace pystencils {

namespace internal_collidesweepthermalized_collidesweepthermalized {
static FUNC_PREFIX void collidesweepthermalized_collidesweepthermalized(
    double *RESTRICT const _data_force, double *RESTRICT _data_pdfs,
    int64_t const _size_force_0, int64_t const _size_force_1,
    int64_t const _size_force_2, int64_t const _stride_force_0,
    int64_t const _stride_force_1, int64_t const _stride_force_2,
    int64_t const _stride_force_3, int64_t const _stride_pdfs_0,
    int64_t const _stride_pdfs_1, int64_t const _stride_pdfs_2,
    int64_t const _stride_pdfs_3, uint32_t block_offset_0,
    uint32_t block_offset_1, uint32_t block_offset_2, double kT,
    double omega_bulk, double omega_even, double omega_odd, double omega_shear,
    uint32_t seed, uint32_t time_step) {
  const double xi_25 = -omega_bulk;
  const double xi_36 = -omega_shear;
  const double xi_37 = xi_36 + 2.0;
  const double xi_38 = xi_37 * 0.5;
  const double xi_43 = xi_37 * 0.0833333333333333;
  const double xi_48 = xi_37 * 0.166666666666667;
  const double xi_58 = xi_37 * 0.25;
  const double xi_63 = xi_37 * 0.0416666666666667;
  const double xi_90 = 2.4494897427831779;
  const double xi_115 = omega_odd * 0.25;
  const double xi_131 = omega_odd * 0.0833333333333333;
  const double xi_196 = omega_shear * 0.25;
  const double xi_211 = omega_odd * 0.0416666666666667;
  const double xi_213 = omega_odd * 0.125;
  const int64_t rr_0 = 0.0;
  const double xi_120 = rr_0 * 0.166666666666667;
  const double xi_186 = rr_0 * 0.0833333333333333;
  for (int64_t ctr_2 = 0; ctr_2 < _size_force_2; ctr_2 += 1) {
    double *RESTRICT _data_pdfs_20_318 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 18 * _stride_pdfs_3;
    double *RESTRICT _data_force_20_30 = _data_force + _stride_force_2 * ctr_2;
    double *RESTRICT _data_pdfs_20_316 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 16 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_37 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 7 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_311 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 11 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_38 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 8 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_313 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 13 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_39 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 9 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_310 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 10 * _stride_pdfs_3;
    double *RESTRICT _data_force_20_31 =
        _data_force + _stride_force_2 * ctr_2 + _stride_force_3;
    double *RESTRICT _data_pdfs_20_35 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 5 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_31 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_315 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 15 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_33 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 3 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_314 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 14 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_32 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 2 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_30 = _data_pdfs + _stride_pdfs_2 * ctr_2;
    double *RESTRICT _data_pdfs_20_34 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 4 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_36 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 6 * _stride_pdfs_3;
    double *RESTRICT _data_pdfs_20_317 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 17 * _stride_pdfs_3;
    double *RESTRICT _data_force_20_32 =
        _data_force + _stride_force_2 * ctr_2 + 2 * _stride_force_3;
    double *RESTRICT _data_pdfs_20_312 =
        _data_pdfs + _stride_pdfs_2 * ctr_2 + 12 * _stride_pdfs_3;
    for (int64_t ctr_1 = 0; ctr_1 < _size_force_1; ctr_1 += 1) {
      double *RESTRICT _data_pdfs_20_318_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_318;
      double *RESTRICT _data_force_20_30_10 =
          _stride_force_1 * ctr_1 + _data_force_20_30;
      double *RESTRICT _data_pdfs_20_316_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_316;
      double *RESTRICT _data_pdfs_20_37_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_37;
      double *RESTRICT _data_pdfs_20_311_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_311;
      double *RESTRICT _data_pdfs_20_38_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_38;
      double *RESTRICT _data_pdfs_20_313_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_313;
      double *RESTRICT _data_pdfs_20_39_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_39;
      double *RESTRICT _data_pdfs_20_310_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_310;
      double *RESTRICT _data_force_20_31_10 =
          _stride_force_1 * ctr_1 + _data_force_20_31;
      double *RESTRICT _data_pdfs_20_35_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_35;
      double *RESTRICT _data_pdfs_20_31_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_31;
      double *RESTRICT _data_pdfs_20_315_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_315;
      double *RESTRICT _data_pdfs_20_33_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_33;
      double *RESTRICT _data_pdfs_20_314_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_314;
      double *RESTRICT _data_pdfs_20_32_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_32;
      double *RESTRICT _data_pdfs_20_30_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_30;
      double *RESTRICT _data_pdfs_20_34_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_34;
      double *RESTRICT _data_pdfs_20_36_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_36;
      double *RESTRICT _data_pdfs_20_317_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_317;
      double *RESTRICT _data_force_20_32_10 =
          _stride_force_1 * ctr_1 + _data_force_20_32;
      double *RESTRICT _data_pdfs_20_312_10 =
          _stride_pdfs_1 * ctr_1 + _data_pdfs_20_312;
      for (int64_t ctr_0 = 0; ctr_0 < _size_force_0; ctr_0 += 1) {
        const double xi_248 = _data_pdfs_20_318_10[_stride_pdfs_0 * ctr_0];
        const double xi_249 = _data_force_20_30_10[_stride_force_0 * ctr_0];
        const double xi_250 = _data_pdfs_20_316_10[_stride_pdfs_0 * ctr_0];
        const double xi_251 = _data_pdfs_20_37_10[_stride_pdfs_0 * ctr_0];
        const double xi_252 = _data_pdfs_20_311_10[_stride_pdfs_0 * ctr_0];
        const double xi_253 = _data_pdfs_20_38_10[_stride_pdfs_0 * ctr_0];
        const double xi_254 = _data_pdfs_20_313_10[_stride_pdfs_0 * ctr_0];
        const double xi_255 = _data_pdfs_20_39_10[_stride_pdfs_0 * ctr_0];
        const double xi_256 = _data_pdfs_20_310_10[_stride_pdfs_0 * ctr_0];
        const double xi_257 = _data_force_20_31_10[_stride_force_0 * ctr_0];
        const double xi_258 = _data_pdfs_20_35_10[_stride_pdfs_0 * ctr_0];
        const double xi_259 = _data_pdfs_20_31_10[_stride_pdfs_0 * ctr_0];
        const double xi_260 = _data_pdfs_20_315_10[_stride_pdfs_0 * ctr_0];
        const double xi_261 = _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0];
        const double xi_262 = _data_pdfs_20_314_10[_stride_pdfs_0 * ctr_0];
        const double xi_263 = _data_pdfs_20_32_10[_stride_pdfs_0 * ctr_0];
        const double xi_264 = _data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0];
        const double xi_265 = _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0];
        const double xi_266 = _data_pdfs_20_36_10[_stride_pdfs_0 * ctr_0];
        const double xi_267 = _data_pdfs_20_317_10[_stride_pdfs_0 * ctr_0];
        const double xi_268 = _data_force_20_32_10[_stride_force_0 * ctr_0];
        const double xi_269 = _data_pdfs_20_312_10[_stride_pdfs_0 * ctr_0];

        double random_7_0;
        double random_7_1;
        philox_double2(time_step, block_offset_0 + ctr_0,
                       block_offset_1 + ctr_1, block_offset_2 + ctr_2, 7, seed,
                       random_7_0, random_7_1);

        double random_6_0;
        double random_6_1;
        philox_double2(time_step, block_offset_0 + ctr_0,
                       block_offset_1 + ctr_1, block_offset_2 + ctr_2, 6, seed,
                       random_6_0, random_6_1);

        double random_5_0;
        double random_5_1;
        philox_double2(time_step, block_offset_0 + ctr_0,
                       block_offset_1 + ctr_1, block_offset_2 + ctr_2, 5, seed,
                       random_5_0, random_5_1);

        double random_4_0;
        double random_4_1;
        philox_double2(time_step, block_offset_0 + ctr_0,
                       block_offset_1 + ctr_1, block_offset_2 + ctr_2, 4, seed,
                       random_4_0, random_4_1);

        double random_3_0;
        double random_3_1;
        philox_double2(time_step, block_offset_0 + ctr_0,
                       block_offset_1 + ctr_1, block_offset_2 + ctr_2, 3, seed,
                       random_3_0, random_3_1);

        double random_2_0;
        double random_2_1;
        philox_double2(time_step, block_offset_0 + ctr_0,
                       block_offset_1 + ctr_1, block_offset_2 + ctr_2, 2, seed,
                       random_2_0, random_2_1);

        double random_1_0;
        double random_1_1;
        philox_double2(time_step, block_offset_0 + ctr_0,
                       block_offset_1 + ctr_1, block_offset_2 + ctr_2, 1, seed,
                       random_1_0, random_1_1);

        double random_0_0;
        double random_0_1;
        philox_double2(time_step, block_offset_0 + ctr_0,
                       block_offset_1 + ctr_1, block_offset_2 + ctr_2, 0, seed,
                       random_0_0, random_0_1);

        const double xi_0 = xi_248 + xi_262;
        const double xi_1 = xi_0 + xi_265;
        const double xi_2 = xi_252 + xi_259 + xi_260;
        const double xi_3 = xi_258 + xi_269;
        const double xi_4 = xi_255 + xi_261;
        const double xi_5 = xi_250 + xi_263;
        const double xi_6 = xi_266 + xi_267;
        const double xi_8 = -xi_255;
        const double xi_9 = -xi_251 + xi_8;
        const double xi_10 = -xi_267;
        const double xi_11 = -xi_254;
        const double xi_12 = -xi_261;
        const double xi_13 = xi_10 + xi_11 + xi_12;
        const double xi_14 = -xi_263;
        const double xi_15 = -xi_256;
        const double xi_16 = xi_14 + xi_15;
        const double xi_17 = -xi_250;
        const double xi_18 = -xi_269;
        const double xi_19 = xi_17 + xi_18;
        const double xi_20 = -xi_248;
        const double xi_21 = xi_10 + xi_20;
        const double xi_22 = -xi_260;
        const double xi_23 = -xi_266;
        const double xi_24 = xi_17 + xi_22 + xi_23 + xi_252;
        const double xi_42 = xi_257 * 0.166666666666667;
        const double xi_50 = xi_249 * 0.166666666666667;
        const double xi_54 = xi_268 * 0.166666666666667;
        const double xi_57 = xi_257 * 0.5;
        const double xi_61 = xi_249 * 0.0833333333333333;
        const double xi_65 = xi_257 * 0.0833333333333333;
        const double xi_75 = xi_268 * 0.0833333333333333;
        const double xi_93 = -xi_264;
        const double xi_94 = xi_258 * 3.0 + xi_266 * 3.0 + xi_93;
        const double xi_95 =
            omega_even * (xi_250 * -3.0 + xi_252 * -3.0 + xi_259 * 3.0 +
                          xi_260 * -3.0 + xi_263 * 3.0 + xi_269 * -3.0 + xi_94);
        const double xi_96 =
            xi_250 * 2.0 + xi_252 * 2.0 + xi_260 * 2.0 + xi_269 * 2.0;
        const double xi_97 = xi_261 * 5.0 + xi_265 * 5.0 + xi_96;
        const double xi_98 =
            omega_even *
            (xi_248 * -5.0 + xi_254 * -5.0 + xi_259 * -2.0 + xi_262 * -5.0 +
             xi_263 * -2.0 + xi_267 * -5.0 + xi_94 + xi_97);
        const double xi_101 = -xi_252;
        const double xi_102 = xi_101 + xi_18;
        const double xi_103 = -xi_253;
        const double xi_106 = -xi_262;
        const double xi_107 = xi_106 + xi_11 + xi_15 + xi_21;
        const double xi_109 = xi_254 * 2.0;
        const double xi_110 = xi_262 * 2.0;
        const double xi_111 = xi_248 * 2.0 + xi_267 * 2.0;
        const double xi_112 =
            omega_even *
            (xi_109 + xi_110 + xi_111 + xi_251 * -7.0 + xi_253 * -7.0 +
             xi_255 * -7.0 + xi_256 * -7.0 + xi_258 * -4.0 + xi_259 * 5.0 +
             xi_263 * 5.0 + xi_266 * -4.0 + xi_93 + xi_97);
        const double xi_113 = xi_101 + xi_269;
        const double xi_114 = xi_113 + xi_14 + xi_22 + xi_250 + xi_259;
        const double xi_116 = xi_114 * xi_115;
        const double xi_118 = xi_103 + xi_256;
        const double xi_122 = random_5_1 - 0.5;
        const double xi_127 = xi_251 * 2.0;
        const double xi_128 = xi_256 * 2.0;
        const double xi_129 = xi_253 * -2.0 + xi_255 * 2.0;
        const double xi_130 = -xi_127 + xi_128 + xi_129 + xi_14 + xi_19 + xi_2;
        const double xi_132 = xi_130 * xi_131;
        const double xi_133 = random_3_0 - 0.5;
        const double xi_138 = random_0_1 - 0.5;
        const double xi_142 = xi_254 + xi_267;
        const double xi_156 = xi_106 + xi_254;
        const double xi_157 = xi_12 + xi_156 + xi_20 + xi_265 + xi_267;
        const double xi_158 = xi_115 * xi_157;
        const double xi_159 = random_4_1 - 0.5;
        const double xi_161 = xi_1 + xi_127 - xi_128 + xi_129 + xi_13;
        const double xi_162 = xi_131 * xi_161;
        const double xi_163 = random_4_0 - 0.5;
        const double xi_168 = xi_250 + xi_260;
        const double xi_169 = xi_102 + xi_168 + xi_23 + xi_258;
        const double xi_170 = xi_115 * xi_169;
        const double xi_173 = random_5_0 - 0.5;
        const double xi_175 = -xi_109 - xi_110 + xi_111 + xi_24 + xi_3;
        const double xi_176 = xi_131 * xi_175;
        const double xi_177 = random_3_1 - 0.5;
        const double xi_184 = xi_112 * 0.0138888888888889;
        const double xi_205 = xi_98 * -0.00714285714285714;
        const double xi_207 = xi_95 * 0.025;
        const double xi_212 = xi_175 * xi_211;
        const double xi_214 = xi_169 * xi_213;
        const double xi_223 = xi_130 * xi_211;
        const double xi_224 = xi_114 * xi_213;
        const double xi_232 = xi_98 * 0.0178571428571429;
        const double xi_238 = xi_157 * xi_213;
        const double xi_239 = xi_161 * xi_211;
        const double vel0Term = xi_1 + xi_253 + xi_256;
        const double vel1Term = xi_2 + xi_251;
        const double vel2Term = xi_254 + xi_3;
        const double rho =
            vel0Term + vel1Term + vel2Term + xi_264 + xi_4 + xi_5 + xi_6;
        const double xi_7 = 1 / (rho);
        const double xi_86 = kT * rho;
        const double xi_87 =
            sqrt(xi_86 * (-((-omega_even + 1.0) * (-omega_even + 1.0)) + 1.0));
        const double xi_88 = xi_87 * (random_6_0 - 0.5) * 3.7416573867739413;
        const double xi_89 = xi_87 * (random_7_0 - 0.5) * 5.4772255750516612;
        const double xi_91 =
            xi_90 * sqrt(xi_86 * (-((xi_25 + 1.0) * (xi_25 + 1.0)) + 1.0)) *
            (random_2_1 - 0.5);
        const double xi_92 = xi_87 * (random_6_1 - 0.5) * 8.3666002653407556;
        const double xi_123 =
            sqrt(xi_86 * (-((-omega_odd + 1.0) * (-omega_odd + 1.0)) + 1.0));
        const double xi_124 = xi_123 * 1.4142135623730951;
        const double xi_125 = xi_124 * 0.5;
        const double xi_126 = xi_122 * xi_125;
        const double xi_134 = xi_123 * xi_90;
        const double xi_135 = xi_134 * 0.166666666666667;
        const double xi_136 = xi_133 * xi_135;
        const double xi_137 = -xi_132 - xi_136;
        const double xi_139 =
            sqrt(xi_86 * (-((xi_36 + 1.0) * (xi_36 + 1.0)) + 1.0));
        const double xi_140 = xi_139 * 0.5;
        const double xi_141 = xi_138 * xi_140;
        const double xi_146 =
            xi_112 * -0.0198412698412698 + xi_88 * -0.119047619047619;
        const double xi_148 = xi_139 * (random_0_0 - 0.5) * 1.7320508075688772;
        const double xi_152 = xi_132 + xi_136;
        const double xi_160 = xi_125 * xi_159;
        const double xi_164 = xi_135 * xi_163;
        const double xi_165 = xi_162 + xi_164;
        const double xi_167 = -xi_162 - xi_164;
        const double xi_174 = xi_125 * xi_173;
        const double xi_178 = xi_135 * xi_177;
        const double xi_179 = -xi_176 - xi_178;
        const double xi_181 = xi_176 + xi_178;
        const double xi_182 = xi_138 * xi_139 * 0.25;
        const double xi_185 = xi_88 * 0.0833333333333333;
        const double xi_195 = xi_140 * (random_1_0 - 0.5);
        const double xi_204 = xi_140 * (random_2_0 - 0.5);
        const double xi_208 = xi_92 * -0.0142857142857143;
        const double xi_209 = xi_89 * 0.05;
        const double xi_215 = xi_134 * 0.0833333333333333;
        const double xi_216 = xi_177 * xi_215;
        const double xi_217 = xi_124 * 0.25;
        const double xi_218 = xi_173 * xi_217;
        const double xi_220 =
            xi_112 * -0.00396825396825397 + xi_88 * -0.0238095238095238;
        const double xi_225 = xi_133 * xi_215;
        const double xi_226 = xi_122 * xi_217;
        const double xi_230 = -xi_182;
        const double xi_233 = xi_92 * 0.0357142857142857;
        const double xi_235 = xi_140 * (random_1_1 - 0.5);
        const double xi_240 = xi_159 * xi_217;
        const double xi_241 = xi_163 * xi_215;
        const double u_0 = xi_7 * (vel0Term + xi_13 + xi_9);
        const double xi_26 = u_0 * xi_249;
        const double xi_27 = xi_26 * 0.333333333333333;
        const double xi_33 = -xi_27;
        const double xi_99 = rho * (u_0 * u_0);
        const double xi_153 = rho * u_0;
        const double xi_154 = -vel0Term + xi_142 + xi_153 + xi_251 + xi_4;
        const double xi_155 = xi_120 * xi_154;
        const double xi_191 = xi_154 * xi_186;
        const double u_1 = xi_7 * (vel1Term + xi_16 + xi_19 + xi_253 + xi_8);
        const double xi_28 = u_1 * xi_257;
        const double xi_29 = xi_28 * 0.333333333333333;
        const double xi_34 = -xi_29;
        const double xi_56 = u_1 * 0.5;
        const double xi_59 = xi_58 * (u_0 * xi_57 + xi_249 * xi_56);
        const double xi_60 = -xi_59;
        const double xi_104 = rho * (u_1 * u_1);
        const double xi_105 = xi_103 + xi_104 + xi_9;
        const double xi_117 = rho * u_1;
        const double xi_119 =
            -vel1Term + xi_117 + xi_118 + xi_255 + xi_269 + xi_5;
        const double xi_121 = xi_119 * xi_120;
        const double xi_187 = xi_119 * xi_186;
        const double xi_197 = xi_196 * (u_0 * xi_117 + xi_118 + xi_251 + xi_8);
        const double xi_198 = -xi_195 - xi_197;
        const double xi_199 = xi_195 + xi_197;
        const double u_2 = xi_7 * (vel2Term + xi_21 + xi_24 + xi_262);
        const double xi_30 = u_2 * xi_268;
        const double xi_31 = xi_30 * 0.333333333333333;
        const double xi_32 = (xi_25 + 2.0) * (xi_27 + xi_29 + xi_31);
        const double xi_35 = xi_30 * 0.666666666666667 + xi_33 + xi_34;
        const double xi_39 = -xi_31;
        const double xi_40 = xi_28 * 0.666666666666667 + xi_33 + xi_39;
        const double xi_41 = xi_26 * 0.666666666666667 + xi_34 + xi_39;
        const double xi_44 = xi_35 * xi_43;
        const double xi_45 = -xi_44;
        const double xi_46 = xi_41 * xi_43;
        const double xi_47 = -xi_46;
        const double xi_49 = xi_40 * xi_48 + xi_45 + xi_47;
        const double xi_51 = xi_40 * xi_43;
        const double xi_52 = -xi_51;
        const double xi_53 = xi_41 * xi_48 + xi_45 + xi_52;
        const double xi_55 = xi_35 * xi_48 + xi_47 + xi_52;
        const double xi_62 = xi_46 - xi_61;
        const double xi_64 = -xi_35 * xi_63;
        const double xi_66 = xi_32 * 0.125;
        const double xi_67 = xi_51 + xi_66;
        const double xi_68 = xi_65 + xi_67;
        const double xi_69 = xi_64 + xi_68;
        const double xi_70 = xi_46 + xi_61;
        const double xi_71 = -xi_65 + xi_67;
        const double xi_72 = xi_64 + xi_71;
        const double xi_73 = xi_58 * (u_2 * xi_57 + xi_268 * xi_56);
        const double xi_74 = -xi_41 * xi_63;
        const double xi_76 = xi_44 + xi_75;
        const double xi_77 = xi_74 + xi_76;
        const double xi_78 = -xi_73;
        const double xi_79 = xi_58 * (u_0 * xi_268 * 0.5 + u_2 * xi_249 * 0.5);
        const double xi_80 = -xi_79;
        const double xi_81 = -xi_40 * xi_63;
        const double xi_82 = xi_66 + xi_76 + xi_81;
        const double xi_83 = xi_44 - xi_75;
        const double xi_84 = xi_74 + xi_83;
        const double xi_85 = xi_66 + xi_81 + xi_83;
        const double xi_100 = rho * (u_2 * u_2);
        const double xi_108 = omega_bulk * (xi_100 + xi_102 + xi_105 + xi_107 +
                                            xi_17 + xi_22 + xi_264 + xi_99);
        const double xi_143 = -xi_100 + xi_258 + xi_266;
        const double xi_144 =
            omega_shear * (xi_0 + xi_105 + xi_142 + xi_143 + xi_16 - xi_259);
        const double xi_145 = xi_144 * 0.125;
        const double xi_147 =
            omega_shear *
            (xi_103 - xi_104 + xi_107 + xi_143 + xi_259 + xi_261 * -2.0 +
             xi_263 + xi_265 * -2.0 + xi_9 + xi_96 + xi_99 * 2.0);
        const double xi_149 =
            xi_147 * -0.0416666666666667 + xi_148 * -0.166666666666667;
        const double xi_150 = xi_149 + xi_89 * -0.1 + xi_95 * -0.05;
        const double xi_151 = xi_141 + xi_145 + xi_146 + xi_150 +
                              xi_92 * 0.0285714285714286 +
                              xi_98 * 0.0142857142857143;
        const double xi_166 =
            xi_146 + xi_147 * 0.0833333333333333 + xi_148 * 0.333333333333333 +
            xi_92 * -0.0714285714285714 + xi_98 * -0.0357142857142857;
        const double xi_171 =
            rho * u_2 - vel2Term + xi_101 + xi_106 + xi_168 + xi_248 + xi_6;
        const double xi_172 = xi_120 * xi_171;
        const double xi_180 = xi_112 * 0.0158730158730159 - xi_141 - xi_145 +
                              xi_150 + xi_88 * 0.0952380952380952 +
                              xi_92 * -0.0428571428571429 +
                              xi_98 * -0.0214285714285714;
        const double xi_183 = xi_144 * 0.0625;
        const double xi_188 =
            xi_108 * 0.0416666666666667 + xi_91 * 0.0833333333333333;
        const double xi_189 = xi_187 + xi_188;
        const double xi_190 =
            xi_152 + xi_182 + xi_183 + xi_184 + xi_185 + xi_189;
        const double xi_192 =
            xi_147 * 0.0208333333333333 + xi_148 * 0.0833333333333333;
        const double xi_193 = -xi_191 + xi_192;
        const double xi_194 = xi_167 + xi_193;
        const double xi_200 = xi_191 + xi_192;
        const double xi_201 = xi_165 + xi_200;
        const double xi_202 = -xi_187 + xi_188;
        const double xi_203 =
            xi_137 + xi_182 + xi_183 + xi_184 + xi_185 + xi_202;
        const double xi_206 = xi_196 * (u_2 * xi_117 + xi_113 + xi_17 + xi_260);
        const double xi_210 =
            xi_149 + xi_204 + xi_205 + xi_206 + xi_207 + xi_208 + xi_209;
        const double xi_219 = xi_171 * xi_186;
        const double xi_221 = xi_219 + xi_220;
        const double xi_222 = -xi_212 + xi_214 - xi_216 + xi_218 + xi_221;
        const double xi_227 = xi_189 - xi_223 + xi_224 - xi_225 + xi_226;
        const double xi_228 = xi_202 + xi_223 - xi_224 + xi_225 - xi_226;
        const double xi_229 =
            xi_149 - xi_204 + xi_205 - xi_206 + xi_207 + xi_208 + xi_209;
        const double xi_231 = -xi_183;
        const double xi_234 =
            xi_181 + xi_188 + xi_221 + xi_230 + xi_231 + xi_232 + xi_233;
        const double xi_236 = xi_196 * (u_2 * xi_153 + xi_10 + xi_156 + xi_248);
        const double xi_237 = -xi_235 - xi_236;
        const double xi_242 = xi_193 - xi_238 + xi_239 - xi_240 + xi_241;
        const double xi_243 = xi_235 + xi_236;
        const double xi_244 = xi_200 + xi_238 - xi_239 + xi_240 - xi_241;
        const double xi_245 = -xi_219 + xi_220;
        const double xi_246 = xi_212 - xi_214 + xi_216 - xi_218 + xi_245;
        const double xi_247 =
            xi_179 + xi_188 + xi_230 + xi_231 + xi_232 + xi_233 + xi_245;
        const double forceTerm_0 =
            xi_32 * -1.5 - xi_35 * xi_38 - xi_38 * xi_40 - xi_38 * xi_41;
        const double forceTerm_1 = xi_42 + xi_49;
        const double forceTerm_2 = -xi_42 + xi_49;
        const double forceTerm_3 = -xi_50 + xi_53;
        const double forceTerm_4 = xi_50 + xi_53;
        const double forceTerm_5 = xi_54 + xi_55;
        const double forceTerm_6 = -xi_54 + xi_55;
        const double forceTerm_7 = xi_60 + xi_62 + xi_69;
        const double forceTerm_8 = xi_59 + xi_69 + xi_70;
        const double forceTerm_9 = xi_59 + xi_62 + xi_72;
        const double forceTerm_10 = xi_60 + xi_70 + xi_72;
        const double forceTerm_11 = xi_68 + xi_73 + xi_77;
        const double forceTerm_12 = xi_71 + xi_77 + xi_78;
        const double forceTerm_13 = xi_62 + xi_80 + xi_82;
        const double forceTerm_14 = xi_70 + xi_79 + xi_82;
        const double forceTerm_15 = xi_68 + xi_78 + xi_84;
        const double forceTerm_16 = xi_71 + xi_73 + xi_84;
        const double forceTerm_17 = xi_62 + xi_79 + xi_85;
        const double forceTerm_18 = xi_70 + xi_80 + xi_85;
        _data_pdfs_20_30_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_0 + xi_108 * -0.5 + xi_112 * 0.0238095238095238 + xi_264 +
            xi_88 * 0.142857142857143 + xi_89 * 0.2 - xi_91 +
            xi_92 * 0.0857142857142857 + xi_95 * 0.1 +
            xi_98 * 0.0428571428571429;
        _data_pdfs_20_31_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_1 - xi_116 + xi_121 - xi_126 + xi_137 + xi_151 + xi_259;
        _data_pdfs_20_32_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_2 + xi_116 - xi_121 + xi_126 + xi_151 + xi_152 + xi_263;
        _data_pdfs_20_33_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_3 - xi_155 + xi_158 + xi_160 + xi_165 + xi_166 + xi_261;
        _data_pdfs_20_34_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_4 + xi_155 - xi_158 - xi_160 + xi_166 + xi_167 + xi_265;
        _data_pdfs_20_35_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_5 - xi_170 + xi_172 - xi_174 + xi_179 + xi_180 + xi_258;
        _data_pdfs_20_36_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_6 + xi_170 - xi_172 + xi_174 + xi_180 + xi_181 + xi_266;
        _data_pdfs_20_37_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_7 + xi_190 + xi_194 + xi_198 + xi_251;
        _data_pdfs_20_38_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_8 + xi_190 + xi_199 + xi_201 + xi_253;
        _data_pdfs_20_39_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_9 + xi_194 + xi_199 + xi_203 + xi_255;
        _data_pdfs_20_310_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_10 + xi_198 + xi_201 + xi_203 + xi_256;
        _data_pdfs_20_311_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_11 + xi_210 + xi_222 + xi_227 + xi_252;
        _data_pdfs_20_312_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_12 + xi_222 + xi_228 + xi_229 + xi_269;
        _data_pdfs_20_313_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_13 + xi_234 + xi_237 + xi_242 + xi_254;
        _data_pdfs_20_314_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_14 + xi_234 + xi_243 + xi_244 + xi_262;
        _data_pdfs_20_315_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_15 + xi_227 + xi_229 + xi_246 + xi_260;
        _data_pdfs_20_316_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_16 + xi_210 + xi_228 + xi_246 + xi_250;
        _data_pdfs_20_317_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_17 + xi_242 + xi_243 + xi_247 + xi_267;
        _data_pdfs_20_318_10[_stride_pdfs_0 * ctr_0] =
            forceTerm_18 + xi_237 + xi_244 + xi_247 + xi_248;
      }
    }
  }
}
} // namespace internal_collidesweepthermalized_collidesweepthermalized

void CollideSweepThermalized::run(IBlock *block) {
  auto force = block->getData<field::GhostLayerField<double, 3>>(forceID);
  auto pdfs = block->getData<field::GhostLayerField<double, 19>>(pdfsID);

  auto &omega_even = this->omega_even_;
  auto block_offset_2 = this->block_offset_2_;
  auto &seed = this->seed_;
  auto &omega_bulk = this->omega_bulk_;
  auto &omega_shear = this->omega_shear_;
  auto &kT = this->kT_;
  auto &time_step = this->time_step_;
  auto &omega_odd = this->omega_odd_;
  auto block_offset_0 = this->block_offset_0_;
  auto block_offset_1 = this->block_offset_1_;
  block_offset_generator(block, block_offset_0, block_offset_1, block_offset_2);
  WALBERLA_ASSERT_GREATER_EQUAL(0, -int_c(force->nrOfGhostLayers()));
  double *RESTRICT const _data_force = force->dataAt(0, 0, 0, 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(0, -int_c(pdfs->nrOfGhostLayers()));
  double *RESTRICT _data_pdfs = pdfs->dataAt(0, 0, 0, 0);
  WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->xSizeWithGhostLayer(),
                                int64_t(cell_idx_c(force->xSize()) + 0));
  const int64_t _size_force_0 = int64_t(cell_idx_c(force->xSize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->ySizeWithGhostLayer(),
                                int64_t(cell_idx_c(force->ySize()) + 0));
  const int64_t _size_force_1 = int64_t(cell_idx_c(force->ySize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->zSizeWithGhostLayer(),
                                int64_t(cell_idx_c(force->zSize()) + 0));
  const int64_t _size_force_2 = int64_t(cell_idx_c(force->zSize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  const int64_t _stride_force_0 = int64_t(force->xStride());
  const int64_t _stride_force_1 = int64_t(force->yStride());
  const int64_t _stride_force_2 = int64_t(force->zStride());
  const int64_t _stride_force_3 = int64_t(1 * int64_t(force->fStride()));
  const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
  const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
  const int64_t _stride_pdfs_2 = int64_t(pdfs->zStride());
  const int64_t _stride_pdfs_3 = int64_t(1 * int64_t(pdfs->fStride()));
  internal_collidesweepthermalized_collidesweepthermalized::
      collidesweepthermalized_collidesweepthermalized(
          _data_force, _data_pdfs, _size_force_0, _size_force_1, _size_force_2,
          _stride_force_0, _stride_force_1, _stride_force_2, _stride_force_3,
          _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2, _stride_pdfs_3,
          block_offset_0, block_offset_1, block_offset_2, kT, omega_bulk,
          omega_even, omega_odd, omega_shear, seed, time_step);
  this->time_step_++;
}

void CollideSweepThermalized::runOnCellInterval(
    const shared_ptr<StructuredBlockStorage> &blocks,
    const CellInterval &globalCellInterval, cell_idx_t ghostLayers,
    IBlock *block) {
  CellInterval ci = globalCellInterval;
  CellInterval blockBB = blocks->getBlockCellBB(*block);
  blockBB.expand(ghostLayers);
  ci.intersect(blockBB);
  blocks->transformGlobalToBlockLocalCellInterval(ci, *block);
  if (ci.empty())
    return;

  auto force = block->getData<field::GhostLayerField<double, 3>>(forceID);
  auto pdfs = block->getData<field::GhostLayerField<double, 19>>(pdfsID);

  auto &omega_even = this->omega_even_;
  auto block_offset_2 = this->block_offset_2_;
  auto &seed = this->seed_;
  auto &omega_bulk = this->omega_bulk_;
  auto &omega_shear = this->omega_shear_;
  auto &kT = this->kT_;
  auto &time_step = this->time_step_;
  auto &omega_odd = this->omega_odd_;
  auto block_offset_0 = this->block_offset_0_;
  auto block_offset_1 = this->block_offset_1_;
  block_offset_generator(block, block_offset_0, block_offset_1, block_offset_2);
  WALBERLA_ASSERT_GREATER_EQUAL(ci.xMin(), -int_c(force->nrOfGhostLayers()));
  WALBERLA_ASSERT_GREATER_EQUAL(ci.yMin(), -int_c(force->nrOfGhostLayers()));
  WALBERLA_ASSERT_GREATER_EQUAL(ci.zMin(), -int_c(force->nrOfGhostLayers()));
  double *RESTRICT const _data_force =
      force->dataAt(ci.xMin(), ci.yMin(), ci.zMin(), 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(ci.xMin(), -int_c(pdfs->nrOfGhostLayers()));
  WALBERLA_ASSERT_GREATER_EQUAL(ci.yMin(), -int_c(pdfs->nrOfGhostLayers()));
  WALBERLA_ASSERT_GREATER_EQUAL(ci.zMin(), -int_c(pdfs->nrOfGhostLayers()));
  double *RESTRICT _data_pdfs =
      pdfs->dataAt(ci.xMin(), ci.yMin(), ci.zMin(), 0);
  WALBERLA_ASSERT_EQUAL(pdfs->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->xSizeWithGhostLayer(),
                                int64_t(cell_idx_c(ci.xSize()) + 0));
  const int64_t _size_force_0 = int64_t(cell_idx_c(ci.xSize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->ySizeWithGhostLayer(),
                                int64_t(cell_idx_c(ci.ySize()) + 0));
  const int64_t _size_force_1 = int64_t(cell_idx_c(ci.ySize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  WALBERLA_ASSERT_GREATER_EQUAL(force->zSizeWithGhostLayer(),
                                int64_t(cell_idx_c(ci.zSize()) + 0));
  const int64_t _size_force_2 = int64_t(cell_idx_c(ci.zSize()) + 0);
  WALBERLA_ASSERT_EQUAL(force->layout(), field::fzyx);
  const int64_t _stride_force_0 = int64_t(force->xStride());
  const int64_t _stride_force_1 = int64_t(force->yStride());
  const int64_t _stride_force_2 = int64_t(force->zStride());
  const int64_t _stride_force_3 = int64_t(1 * int64_t(force->fStride()));
  const int64_t _stride_pdfs_0 = int64_t(pdfs->xStride());
  const int64_t _stride_pdfs_1 = int64_t(pdfs->yStride());
  const int64_t _stride_pdfs_2 = int64_t(pdfs->zStride());
  const int64_t _stride_pdfs_3 = int64_t(1 * int64_t(pdfs->fStride()));
  internal_collidesweepthermalized_collidesweepthermalized::
      collidesweepthermalized_collidesweepthermalized(
          _data_force, _data_pdfs, _size_force_0, _size_force_1, _size_force_2,
          _stride_force_0, _stride_force_1, _stride_force_2, _stride_force_3,
          _stride_pdfs_0, _stride_pdfs_1, _stride_pdfs_2, _stride_pdfs_3,
          block_offset_0, block_offset_1, block_offset_2, kT, omega_bulk,
          omega_even, omega_odd, omega_shear, seed, time_step);
  this->time_step_++;
}

} // namespace pystencils
} // namespace walberla

#if (defined WALBERLA_CXX_COMPILER_IS_GNU) ||                                  \
    (defined WALBERLA_CXX_COMPILER_IS_CLANG)
#pragma GCC diagnostic pop
#endif

#if (defined WALBERLA_CXX_COMPILER_IS_INTEL)
#pragma warning pop
#endif