#ifndef EGG_MODEL_INLINE_HPP
#define EGG_MODEL_INLINE_HPP

#include "config.hpp"

#ifdef EGG_MODEL

#include "Particle.hpp"
#include "random.hpp"
#include "rotation.hpp"
#include "thermostat.hpp"

#include <utils/Vector.hpp>

#include <cmath>

// called from VirtualSitesRelative::update()
inline void egg_model_update_space_quats(Particle &p) {

      p.quat() = p_ref.quat() * p.vs_relative().quat;
      p.egg_model_params().axis_quat_space_fixed = p_ref.quat() * p.egg_model_params().axis_quat_body_fixed;
}


// called from force_calc()
inline void egg_model_calc_internal_magnetic_torque(Particle &p) {
    
        Utils::Vector3d vec_e = p.calc_director();
        Utils::Vector3d vec_n = p.calc_axis();

        auto const torque = 2 * p.aniso_energy() * vector_product(vec_e,vec_n) * (vec_e * vec_n);

        torque = convert_vector_space_to_body(p, torque);

        // torque = mask(p.rotation(), torque);

        p.internal_magnetic_torque() = torque;
        
}

// called from brownian_dynamics_propagator()
inline void egg_model_bd_internal_rotation(BrownianThermostat const &brownian, Particle &p, double dt, double kT) {

  Utils::Vector3d dphi = {};

  auto const noise = Random::noise_gaussian<RNGSalt::BROWNIAN_ROT_INC>(brownian.rng_counter(), brownian.rng_seed(), p.id());

  for (int j = 0; j < 3; j++) {

    // if (p.can_rotate_around(j)) {

     dphi[j] = (p.torque()[j] + p.internal_magnetic_torque()[j]) * dt / p.egg_gamma() + noise[j] * sqrt(2 * dt * kT / p.egg_gamma());

     // }

  }

  // dphi = mask(p.rotation(), dphi);

  double dphi_m = dphi.norm();
  if (dphi_m != 0.) {
    auto const dphi_u = dphi / dphi_m;
    
    if (std::abs(dphi_m) > std::numeric_limits<double>::epsilon())
      // p.vs_relative().quat = p.vs_relative().quat * boost::qvm::rot_quat(mask(p.rotation(), dphi_u), phi);
      p.vs_relative().quat = p.vs_relative().quat * boost::qvm::rot_quat(dphi_u, dphi_m);
  }

}

#endif // EGG_MODEL
#endif // EGG_MODEL_INLINE_HPP