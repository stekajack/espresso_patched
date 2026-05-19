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

#if defined(ESPRESSO_THERMAL_STONER_WOHLFARTH) || defined(ESPRESSO_EGG_MODEL) || defined(ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET)
#include "ParticleMagnetodynamics.hpp"

#include "ParticleHandle.hpp"

#include "script_interface/get_value.hpp"

#include "core/cell_system/CellStructure.hpp"

#include <utils/quaternion.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace ScriptInterface {
namespace Particles {

namespace {

#ifdef ESPRESSO_EGG_MODEL
static auto quat2vector(Utils::Quaternion<double> const &q) {
  return Utils::Vector4d{{q[0], q[1], q[2], q[3]}};
}

static auto get_quaternion_safe(std::string const &name, Variant const &value) {
  auto const q = get_value<Utils::Vector4d>(value);
  if (q.norm2() == 0.) {
    throw std::domain_error(error_msg(name, "must be non-zero"));
  }
  return Utils::Quaternion<double>{{q[0], q[1], q[2], q[3]}};
}
#endif

static void validate_unique_model(Particle const &p) {
  if (p.enabled_magnetodynamics_models() > 1) {
    throw std::runtime_error(
        "A particle can only enable one magnetodynamics model at a time");
  }
}

#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
static void update_tsw(ParticleMagnetodynamicsParameters &magnetodynamics,
                       VariantMap const &dict) {
  if (dict.contains("is_enabled")) {
    magnetodynamics.tsw.is_enabled = get_value<bool>(dict.at("is_enabled"));
  }
  if (dict.contains("sw_phi_0")) {
    magnetodynamics.tsw.phi0 = get_value<double>(dict.at("sw_phi_0"));
  }
  if (dict.contains("sat_mag")) {
    magnetodynamics.tsw.sat_mag = get_value<double>(dict.at("sat_mag"));
  }
  if (dict.contains("anisotropy_field_inv")) {
    magnetodynamics.tsw.ani_fld_inv =
        get_value<double>(dict.at("anisotropy_field_inv"));
  }
  if (dict.contains("anisotropy_energy")) {
    magnetodynamics.tsw.ani_energy =
        get_value<double>(dict.at("anisotropy_energy"));
  }
  if (dict.contains("sw_tau0_inv")) {
    magnetodynamics.tsw.tau0_inv =
        get_value<double>(dict.at("sw_tau0_inv"));
  }
  if (dict.contains("sw_dt_incr")) {
    magnetodynamics.tsw.dt_incr = get_value<double>(dict.at("sw_dt_incr"));
  }
}
#endif

#ifdef ESPRESSO_EGG_MODEL
static void update_egg(ParticleMagnetodynamicsParameters &magnetodynamics,
                       VariantMap const &dict) {
  if (dict.contains("is_enabled")) {
    magnetodynamics.egg.is_enabled = get_value<bool>(dict.at("is_enabled"));
  }
  if (dict.contains("gamma")) {
    auto const gamma = get_value<double>(dict.at("gamma"));
    if (gamma <= 0.) {
      throw std::domain_error(
          error_msg("magnetodynamics.egg", "gamma must be a float > 0"));
    }
    magnetodynamics.egg.gamma = gamma;
  }
  if (dict.contains("anisotropy_energy")) {
    magnetodynamics.egg.anisotropy_energy =
        get_value<double>(dict.at("anisotropy_energy"));
  }
  if (dict.contains("axis_quat_body")) {
    magnetodynamics.egg.axis_quat_body =
        get_quaternion_safe("axis_quat_body", dict.at("axis_quat_body"));
  }
}
#endif

#ifdef ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET
static void update_ideal(ParticleMagnetodynamicsParameters &magnetodynamics,
                         VariantMap const &dict) {
  if (dict.contains("is_enabled")) {
    magnetodynamics.ideal.is_enabled = get_value<bool>(dict.at("is_enabled"));
  }
  if (dict.contains("sat_mag")) {
    auto const sat_mag = get_value<double>(dict.at("sat_mag"));
    if (sat_mag <= 0.) {
      throw std::domain_error(
          error_msg("magnetodynamics.ideal", "sat_mag must be a float > 0"));
    }
    magnetodynamics.ideal.sat_mag = sat_mag;
  }
}
#endif

} // namespace

ParticleMagnetodynamics::ParticleMagnetodynamics() {
  add_parameters({
#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
      {"tsw",
       [this](Variant const &value) {
         auto const dict = get_value<VariantMap>(value);
         auto &system = *get_system();
         auto &cell_structure = *system.cell_structure;
         auto const &comm = context()->get_comm();
         auto const ptr = get_real_particle(comm, m_pid, cell_structure);
         if (ptr != nullptr) {
           auto updated = ptr->magnetodynamics();
           update_tsw(updated, dict);
           auto copy = *ptr;
           copy.magnetodynamics() = updated;
           validate_unique_model(copy);
           ptr->magnetodynamics() = updated;
         }
         system.on_particle_change();
       },
       [this]() {
         auto &system = *get_system();
         auto &cell_structure = *system.cell_structure;
         auto const &comm = context()->get_comm();
         auto const *ptr = const_cast<Particle const *>(
             get_real_particle(comm, m_pid, cell_structure));
         if (ptr == nullptr) {
           return VariantMap{};
         }
         auto const &p = *ptr;
         return VariantMap{{"is_enabled", p.stoner_wohlfarth_is_enabled()},
                           {"sw_phi_0", p.stoner_wohlfarth_phi_0()},
                           {"sat_mag",
                            p.stoner_wohlfarth_saturation_magnetization()},
                           {"anisotropy_field_inv",
                            p.magnetic_anisotropy_field_inv()},
                           {"anisotropy_energy",
                            p.magnetic_anisotropy_energy()},
                           {"sw_tau0_inv", p.stoner_wohlfarth_tau0_inv()},
                           {"sw_dt_incr", p.stoner_wohlfarth_dt_incr()}};
       }},
#endif
#ifdef ESPRESSO_EGG_MODEL
      {"egg",
       [this](Variant const &value) {
         auto const dict = get_value<VariantMap>(value);
         auto &system = *get_system();
         auto &cell_structure = *system.cell_structure;
         auto const &comm = context()->get_comm();
         auto const ptr = get_real_particle(comm, m_pid, cell_structure);
         if (ptr != nullptr) {
           auto updated = ptr->magnetodynamics();
           update_egg(updated, dict);
           auto copy = *ptr;
           copy.magnetodynamics() = updated;
           validate_unique_model(copy);
           ptr->magnetodynamics() = updated;
         }
         system.on_particle_change();
       },
       [this]() {
         auto &system = *get_system();
         auto &cell_structure = *system.cell_structure;
         auto const &comm = context()->get_comm();
         auto const *ptr = const_cast<Particle const *>(
             get_real_particle(comm, m_pid, cell_structure));
         if (ptr == nullptr) {
           return VariantMap{};
         }
         auto const &p = *ptr;
         return VariantMap{{"is_enabled", p.egg_model_is_enabled()},
                           {"gamma", p.egg_model_gamma()},
                           {"anisotropy_energy",
                            p.egg_model_anisotropy_energy()},
                           {"axis_quat_body",
                            quat2vector(p.egg_model_axis_quat_body())},
                           {"axis", p.egg_model_axis()}};
       }},
#endif
#ifdef ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET
      {"ideal",
       [this](Variant const &value) {
         auto const dict = get_value<VariantMap>(value);
         auto &system = *get_system();
         auto &cell_structure = *system.cell_structure;
         auto const &comm = context()->get_comm();
         auto const ptr = get_real_particle(comm, m_pid, cell_structure);
         if (ptr != nullptr) {
           auto updated = ptr->magnetodynamics();
           update_ideal(updated, dict);
           auto copy = *ptr;
           copy.magnetodynamics() = updated;
           validate_unique_model(copy);
           ptr->magnetodynamics() = updated;
         }
         system.on_particle_change();
       },
       [this]() {
         auto &system = *get_system();
         auto &cell_structure = *system.cell_structure;
         auto const &comm = context()->get_comm();
         auto const *ptr = const_cast<Particle const *>(
             get_real_particle(comm, m_pid, cell_structure));
         if (ptr == nullptr) {
           return VariantMap{};
         }
         auto const &p = *ptr;
         return VariantMap{
             {"is_enabled", p.ideal_magnetizable_superparamagnet_is_enabled()},
             {"sat_mag",
              p.ideal_magnetizable_superparamagnet_saturation_magnetization()},
         };
       }},
#endif
  });
}

void ParticleMagnetodynamics::do_construct(VariantMap const &params) {
  m_pid = get_value<int>(params, "id");
}

void ParticleMagnetodynamics::apply_nested_update(VariantMap const &params) const {
  auto &system = *get_system();
  auto &cell_structure = *system.cell_structure;
  auto const &comm = context()->get_comm();
  auto const ptr = get_real_particle(comm, m_pid, cell_structure);
  if (ptr != nullptr) {
    auto updated = ptr->magnetodynamics();
    for (auto const &[key, value] : params) {
      auto const model = std::string{key};
      if (not has_parameter(model)) {
        throw UnknownParameter(model);
      }
      auto const dict = get_value<VariantMap>(value);
#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
      if (model == "tsw") {
        update_tsw(updated, dict);
        continue;
      }
#endif
#ifdef ESPRESSO_EGG_MODEL
      if (model == "egg") {
        update_egg(updated, dict);
        continue;
      }
#endif
#ifdef ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET
      if (model == "ideal") {
        update_ideal(updated, dict);
        continue;
      }
#endif
    }
    auto copy = *ptr;
    copy.magnetodynamics() = updated;
    validate_unique_model(copy);
    ptr->magnetodynamics() = updated;
  }
  system.on_particle_change();
}

} // namespace Particles
} // namespace ScriptInterface
#endif
