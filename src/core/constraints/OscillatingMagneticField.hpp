
#ifndef CONSTRAINTS_OSCILLATINGMAGNETICFIELD_HPP
#define CONSTRAINTS_OSCILLATINGMAGNETICFIELD_HPP
/** \file OscillatingMagneticField.hpp
 *  Routines to calculate forses and energy for  AC Magnetic field.
 *  Field is described by vector-valued direction \vec{m}, magnitude,
 *  frequrecy and phase shift, so
 *  H = magnitude*cos(frequecy*time + phase_shift) \vec{m}
 *
 */
#include "Constraint.hpp"
#include "Particle.hpp"

namespace Constraints {

class OscillatingMagneticField : public Constraint {
public:
  OscillatingMagneticField()
      : m_direction({0., 0., 1.}), m_magnitude(0.), m_frequency(0.),
        m_phase_shift(0) {}

  void set_direction(Utils::Vector3d const &direction) {
    m_direction = direction;
    m_direction.normalize();
  }

  void set_magnitude(const double &magnitude) { m_magnitude = magnitude; }

  void set_phase_shift(const double &shift) { m_phase_shift = shift; }

  void set_frequency(const double frequency) { m_frequency = frequency; }

  Utils::Vector3d &direction() { return m_direction; }

  /** Latest evaluated field vector (updated when force/add_energy is called). */
  Utils::Vector3d H() const { return m_last_field; }

  /** Compute field at simulation time t. */
  Utils::Vector3d field_at(double t) const { return field(t); }

  /** Simulation time corresponding to the cached field. */
  double last_time() const { return m_last_time; }

  double &phase_shift() { return m_phase_shift; }

  double &frequency() { return m_frequency; }

  double &magnitude() { return m_magnitude; }

  void add_energy(const Particle &p, const Utils::Vector3d &, double t,
                  Observable_stat &energy) const override;

  ParticleForce force(const Particle &p, const Utils::Vector3d &,
                      double t) override;

  bool fits_in_box(Utils::Vector3d const &box) const override { return true; }

private:
  Utils::Vector3d field(double t) const {
    auto const factor = std::cos(m_frequency * t + m_phase_shift) * m_magnitude;
    return factor * m_direction;
  }

  Utils::Vector3d m_direction;
  double m_magnitude;
  double m_frequency;
  double m_phase_shift;
  mutable double m_last_time{0.};
  mutable Utils::Vector3d m_last_field{};
};

} /* namespace Constraints */

#endif
