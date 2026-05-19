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

#pragma once

#include "script_interface/auto_parameters/AutoParameters.hpp"

#include "core/system/System.hpp"

#include <cassert>
#include <memory>

namespace ScriptInterface {
namespace Particles {

class ParticleMagnetodynamics
    : public AutoParameters<ParticleMagnetodynamics> {
  int m_pid = -1;
  mutable std::weak_ptr<::System::System> m_system;

  auto get_system() const {
    auto ptr = m_system.lock();
    assert(ptr != nullptr);
    return ptr;
  }

public:
  ParticleMagnetodynamics();

  void do_construct(VariantMap const &params) override;
  void attach(std::weak_ptr<::System::System> system) { m_system = system; }
  void apply_nested_update(VariantMap const &params) const;
};

} // namespace Particles
} // namespace ScriptInterface
