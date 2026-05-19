#pragma once

#include <config/config.hpp>

#include "Particle.hpp"

#include <utils/Vector.hpp>

#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
void stoner_wohlfarth_no_field(Particle &p, Utils::Vector3d const &e_k,
                               double kT, double noise);

void stoner_wohlfarth_main(Particle &p, Utils::Vector3d const &e_k,
                           Utils::Vector3d const &ext_fld_dpl, double kT,
                           double noise);
#endif

#ifdef ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET
void ideal_magnetizable_superparamagnet_langevin(
    Particle &p, Utils::Vector3d const &ext_fld_dpl, double kT);
#endif
