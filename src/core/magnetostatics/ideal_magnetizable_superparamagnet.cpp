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

#ifdef ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET

#include "Particle.hpp"
#include "cell_system/CellStructure.hpp"
#include "cells.hpp"
#include "constraints/Constraints.hpp"
#include "constraints/HomogeneousMagneticField.hpp"
#include "rotation.hpp"
#include "system/System.hpp"
#include "thermostat.hpp"
#include "virtual_sites/relative.hpp"

#include <utils/Vector.hpp>

#include <cmath>
#include <memory>

// absolute error precision required for the optimiser
constexpr static double dipm_equals_zero2 = std::pow(1e-5, 2);

/**
 * @brief Langevin magnetization function.
 *
 * @param p Particle to magnetize.
 */
void ideal_magnetizable_superparamagnet_langevin(Particle &p, Utils::Vector3d const &ext_fld,
                                 double const kT) {
  auto const dipm_saturated = p.ideal_magnetizable_superparamagnet_saturation_magnetization();

  auto const tri2 = ext_fld.norm2();
  if (tri2 < dipm_equals_zero2) {
    p.dipm() = 0.;
    return;
  }
  auto const tri = std::sqrt(tri2);

  auto const alpha = dipm_saturated * tri / kT;

  double L;
  if (alpha < 1e-8) {
    // small-x expansion to avoid numerical instability
    L = alpha / 3.;
  } else {
    L = 1.0 / std::tanh(alpha) - 1.0 / alpha;
  }

  auto const dip_new = (dipm_saturated * L / tri) * ext_fld;
  auto const [quat, dipm_new] = convert_dip_to_quat(dip_new);

  p.dipm() = dipm_new;
  p.quat() = quat;
}


#endif // ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET
