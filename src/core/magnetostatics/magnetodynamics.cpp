/*
 * Copyright (C) 2026 The ESPResSo project
 *
 * This file is part of ESPResSo.
 *
 * ESPRESSO is free software: you can redistribute it and/or modify
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

#if defined(ESPRESSO_THERMAL_STONER_WOHLFARTH) || defined(ESPRESSO_EGG_MODEL) || defined(ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET)

#include "Particle.hpp"
#include "PropagationMode.hpp"
#include "cell_system/CellStructure.hpp"
#include "constraints/Constraints.hpp"
#include "constraints/HomogeneousMagneticField.hpp"
#include "constraints/OscillatingMagneticField.hpp"
#include "errorhandling.hpp"
#include "integrators/Propagation.hpp"
#include "magnetodynamics.hpp"
#include "rotation.hpp"
#include "system/System.hpp"
#include "thermostat.hpp"
#include "virtual_sites/relative.hpp"

#include <utils/Vector.hpp>

#include <cmath>
#include <memory>

#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
#include "random.hpp"

#include <utils/uniform.hpp>
#endif

static auto get_external_field(Constraints::Constraints const &constraints,
                               double time) {
  using HomogeneousMagneticField = ::Constraints::HomogeneousMagneticField;
  using OscillatingMagneticField = ::Constraints::OscillatingMagneticField;
  Utils::Vector3d ext_fld = {0., 0., 0.};
  for (auto const &constraint : constraints) {
    if (auto ptr =
            std::dynamic_pointer_cast<HomogeneousMagneticField>(constraint)) {
      ext_fld += ptr->H();
    }
    if (auto osc_ptr =
            std::dynamic_pointer_cast<OscillatingMagneticField>(constraint)) {
      ext_fld += osc_ptr->field_at(time);
    }
  }
  return ext_fld;
}

void System::System::magnetodynamics_sanity_checks() const {
  for (auto const &p : cell_structure->local_particles()) {
    if (p.enabled_magnetodynamics_models() > 1) {
      runtimeErrorMsg()
          << "Particles can only enable one magnetodynamics model at a time";
      break;
    }
  }

#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
  if (thermostat->thermo_switch == THERMO_OFF) {
    for (auto const &p : cell_structure->local_particles()) {
      if (p.stoner_wohlfarth_is_enabled()) {
        runtimeErrorMsg()
            << "The thermal Stoner-Wohlfarth model requires a thermostat";
        break;
      }
    }
  }
#endif

#ifdef ESPRESSO_EGG_MODEL
  for (auto const &p : cell_structure->local_particles()) {
    using namespace PropagationMode;
    if (not p.egg_model_is_enabled()) {
      continue;
    }
    if (thermostat->thermo_switch != THERMO_BROWNIAN) {
      runtimeErrorMsg() << "The egg model requires the BD thermostat";
      break;
    }
    if (propagation->integ_switch != INTEG_METHOD_BD) {
      runtimeErrorMsg() << "The egg model requires the BD integrator";
      break;
    }
    auto constexpr egg_propagation = TRANS_VS_RELATIVE | ROT_VS_INDEPENDENT;
    if ((p.propagation() & egg_propagation) != egg_propagation) {
      runtimeErrorMsg() << "The egg model requires virtual sites with "
                           "TRANS_VS_RELATIVE and ROT_VS_INDEPENDENT";
      break;
    }
    if (p.vs_relative().to_particle_id == -1) {
      runtimeErrorMsg() << "The egg model requires a virtual site with "
                           "a reference particle";
      break;
    }
  }
#endif

#ifdef ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET
  if (thermostat->thermo_switch == THERMO_OFF) {
    for (auto const &p : cell_structure->local_particles()) {
      if (p.ideal_magnetizable_superparamagnet_is_enabled()) {
        runtimeErrorMsg() << "The ideal magnetizable superparamagnet model requires a thermostat";
        break;
      }
    }
  }
#endif
}

#if defined(ESPRESSO_THERMAL_STONER_WOHLFARTH) || defined(ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET)
void System::System::integrate_magnetodynamics() {
  auto const ext_fld = get_external_field(*constraints, get_sim_time());
  auto const kT = thermostat->kT;
  cell_structure->for_each_local_particle([&](Particle &p) {
    auto const ext_fld_dpl = ext_fld + p.dip_fld();
#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
    if (p.is_virtual() and p.stoner_wohlfarth_is_enabled()) {
      auto *p_ref = get_reference_particle(*cell_structure, p);
      if (not p_ref) {
        return;
      }
      auto const e_k = p_ref->calc_director();
      auto const random_ints =
          Random::philox_4_uint64s<RNGSalt::THERMAL_STONER_WOHLFARTH>(
              thermostat->get_philox_counter(), thermostat->get_philox_seed(),
              p.id());
      auto const noise = Utils::uniform(random_ints[0]);
      if (ext_fld_dpl.norm2() == 0.) {
        stoner_wohlfarth_no_field(p, e_k, kT, noise);
      } else {
        stoner_wohlfarth_main(p, e_k, ext_fld_dpl, kT, noise);
      }
      return;
    }
#endif
#ifdef ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET
    if (p.is_virtual() and p.ideal_magnetizable_superparamagnet_is_enabled()) {
      auto *p_ref = get_reference_particle(*cell_structure, p);
      if (not p_ref) {
        return;
      }
      ideal_magnetizable_superparamagnet_langevin(p, ext_fld_dpl, kT);
    }
#endif
  });
}
#endif

#endif
