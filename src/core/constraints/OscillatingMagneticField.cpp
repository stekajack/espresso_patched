#include "OscillatingMagneticField.hpp"
#include "Observable_stat.hpp"
#include <cmath>

namespace Constraints {

ParticleForce OscillatingMagneticField::force(const Particle &p,
                                              const Utils::Vector3d &,
                                              double t) {
  auto const field_vec = field(t);
  m_last_field = field_vec;
  m_last_time = t;
#if defined(ESPRESSO_ROTATION) && defined(ESPRESSO_DIPOLES)
  return {Utils::Vector3d{}, vector_product(p.calc_dip(), field_vec)};
#else
  return {Utils::Vector3d{}};
#endif
}

void OscillatingMagneticField::add_energy(const Particle &p,
                                          const Utils::Vector3d &, double t,
                                          Observable_stat &energy) const {
  auto const field_vec = field(t);
  m_last_field = field_vec;
  m_last_time = t;
#ifdef ESPRESSO_DIPOLES
  energy.dipolar[0] += -1.0 * field_vec * p.calc_dip();
#endif
}

} // namespace Constraints
