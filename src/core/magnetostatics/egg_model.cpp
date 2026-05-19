/*
 * Copyright (C) 2026 The ESPResSo project
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <config/config.hpp>

#ifdef ESPRESSO_EGG_MODEL

#include "Particle.hpp"
#include "random.hpp"
#include "rotation.hpp"
#include "system/System.hpp"
#include "thermostat.hpp"
#include "virtual_sites/relative.hpp"

#include <utils/Vector.hpp>

#include <cmath>

static void egg_model_update_axis(Particle const &p_ref, Particle &p) {
  p.egg_model_axis_quat_space() =
      p_ref.quat() * p.egg_model_axis_quat_body();
  p.quat() = p_ref.quat() * p.vs_relative().quat;
}

static void egg_model_calc_internal_magnetic_torque(Particle &p) {
  auto const vec_e = p.calc_director();
  auto const vec_n = p.egg_model_axis();

  auto const torque = 2. * p.egg_model_anisotropy_energy() *
                      vector_product(vec_e, vec_n) * (vec_e * vec_n);

  p.egg_model_internal_magnetic_torque() =
      convert_vector_space_to_body(p, torque);
}

static void egg_model_brownian_rotation(Particle const &p_ref,
                                        BrownianThermostat const &brownian,
                                        Particle &p, double dt, double kT) {
  if (not p.can_rotate()) {
    return;
  }

  convert_torque_to_body_frame_apply_fix(p);
  egg_model_calc_internal_magnetic_torque(p);

  auto const noise = Random::noise_gaussian<RNGSalt::BROWNIAN_ROT_INC>(
      brownian.rng_counter(), brownian.rng_seed(), p.id());
  auto const gamma_inv = 1. / p.egg_model_gamma();

  Utils::Vector3d dphi = {};
  for (unsigned int j = 0; j < 3; j++) {
    if (p.can_rotate_around(j)) {
      dphi[j] =
          (p.torque()[j] + p.egg_model_internal_magnetic_torque()[j]) * dt *
              gamma_inv +
          noise[j] * std::sqrt(2. * dt * kT * gamma_inv);
    }
  }

  auto const dphi_m = dphi.norm();
  if (dphi_m != 0.) {
    auto const dphi_u = dphi / dphi_m;
    p.vs_relative().quat =
        p.vs_relative().quat * boost::qvm::rot_quat(dphi_u, dphi_m);
    p.quat() = p_ref.quat() * p.vs_relative().quat;
  }
}

void System::System::integrate_egg_model() {
  auto const kT = thermostat->kT;
  auto const &brownian = *thermostat->brownian;
  cell_structure->for_each_local_particle([&](Particle &p) {
    if (not p.is_virtual() or not p.egg_model_is_enabled()) {
      return;
    }
    auto const *p_ref = get_reference_particle(*cell_structure, p);
    if (not p_ref) {
      return;
    }
    egg_model_update_axis(*p_ref, p);
    egg_model_brownian_rotation(*p_ref, brownian, p, time_step, kT);
  });
}

#endif // ESPRESSO_EGG_MODEL
