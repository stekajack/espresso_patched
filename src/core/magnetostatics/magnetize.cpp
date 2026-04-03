/*
 * Copyright (C) 2025 The ESPResSo project
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

#ifdef ESPRESSO_MAGNETIZE

#include "Particle.hpp"
#include "cell_system/CellStructure.hpp"
#include "cells.hpp"
#include "constraints/Constraints.hpp"
#include "constraints/HomogeneousMagneticField.hpp"
#include "errorhandling.hpp"
#include "random.hpp"
#include "rotation.hpp"
#include "system/System.hpp"
#include "thermostat.hpp"
#include "virtual_sites/relative.hpp"

#include <utils/Vector.hpp>
#include <utils/uniform.hpp>

#include <cassert>
#include <cmath>
#include <memory>
#include <numbers>
#include <utility>
#include <vector>

// absolute error precision required for the optimiser
constexpr static double dipm_equals_zero2 = std::pow(1e-5, 2);

/**
 * @brief Langevin magnetization function.
 *
 * @param p Particle to magnetize.
 * @return Energy value for the given phi.
 */
static void magnetize_p_Langevin(Particle &p, Utils::Vector3d const &ext_fld)
{
  const double dipm_saturated = p.dipm_sat();

  auto const ext_fld_dpl = ext_fld + p.dip_fld();
  const double tri2 = ext_fld_dpl.norm2();
  if (tri2 < dipm_equals_zero2) {
    p.dipm() = 0.;
    return;
  }
  const double tri = std::sqrt(tri2);

  const double alpha = 3 * p.mag_susc_0() / dipm_saturated * tri;

  double L;
  if (alpha < 1e-8) {
    // small-x expansion to avoid numerical instability
    L = alpha / 3.;
  } else {
    L = 1.0 / std::tanh(alpha) - 1.0 / alpha;
  }

  auto const dip_new = (dipm_saturated * L / tri) * ext_fld_dpl;

  auto const [quat, dipm_new] =
      convert_dip_to_quat(dip_new);

  p.dipm() = dipm_new;
  p.quat() = quat;
}

/**
 * @brief Froelich-kennely magnetization function.
 *
 * @param p Particle to magnetize.
 * @return Energy value for the given phi.
 */
static void magnetize_p_froelich_kennelly(Particle &p, Utils::Vector3d const &ext_fld)
{
  const double dipm_saturated = p.dipm_sat();
  const double xi0 = p.mag_susc_0();

  auto const ext_fld_dpl = ext_fld + p.dip_fld();
  const double tri2 = ext_fld_dpl.norm2();
  if (tri2 < dipm_equals_zero2) {
    p.dipm() = 0.;
    return;
  }
  const double tri = std::sqrt(tri2);

  const double pre_froelich = xi0 * dipm_saturated / (dipm_saturated + xi0 * tri);

  auto const dip_new = pre_froelich * ext_fld_dpl;

  auto const [quat, dipm_new] =
      convert_dip_to_quat(dip_new);

  p.dipm() = dipm_new;
  p.quat() = quat;
}

/**
 * @brief Collect external homogeneous magnetic field from active constraints.
 *
 * Iterate over constraints and sum the homogeneous magnetic field vectors
 * provided by @ref Constraints::HomogeneousMagneticField objects.
 *
 * @return The total external homogeneous magnetic field.
 */
static auto get_external_field_testing(Constraints::Constraints const &constraints) {
  using HomogeneousMagneticField = ::Constraints::HomogeneousMagneticField;
  Utils::Vector3d ext_fld = {0., 0., 0.};
  for (auto const &constraint : constraints) {
    auto ptr = std::dynamic_pointer_cast<HomogeneousMagneticField>(constraint);
    if (ptr) {
      ext_fld += ptr->H();
    }
  }
  return ext_fld;
}

/**
 * @brief Run magnetodynamics update for local virtual particles.
 *
 * Iterate over local particles and update the dipole moment of virtual
 * particles according to the chosen magnetization funciton.
 * Collect active homogeneous external magnetic fields from constraints and
 * add the per-particle dipolar contribution before performing dipole update.
 */
void System::System::integrate_magnetodynamics_testing() {
  // collect HomogeneousMagneticFields if active
  auto const ext_fld = get_external_field_testing(*constraints);
  cell_structure->for_each_local_particle([&](Particle &p) {
    if (not p.is_magnetizable()) {
      return;
    }
    auto *p_ref = get_reference_particle(*cell_structure, p);
    if (not p_ref) {
      return;
    }
    switch (p.magnetize_func()) {
      case 0:
        magnetize_p_Langevin(p, ext_fld);
        break;
      case 1:
        magnetize_p_froelich_kennelly(p, ext_fld);
        break;
    }
  });
}

#endif // ESPRESSO_MAGNETIZE
