/*
 * Copyright (C) 2017-2026 The ESPResSo project
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

#define BOOST_TEST_MODULE "Particle struct test"
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <config/config.hpp>

#include "Particle.hpp"
#include "PropagationMode.hpp"

#include <utils/compact_vector.hpp>
#include <utils/serialization/memcpy_archive.hpp>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <algorithm>
#include <array>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

void check_particle_force(ParticleForce const &out, ParticleForce const &ref) {
  BOOST_TEST(out.f == ref.f, boost::test_tools::per_element());
#ifdef ESPRESSO_ROTATION
  BOOST_TEST(out.torque == ref.torque, boost::test_tools::per_element());
#endif
}

BOOST_AUTO_TEST_CASE(comparison) {
  {
    Particle p, q;

    p.id() = 1;
    q.id() = 2;

    BOOST_CHECK(p != q);
    BOOST_CHECK(not(p == q));
  }

  {
    Particle p, q;

    p.id() = 2;
    q.id() = 2;

    BOOST_CHECK(not(p != q));
    BOOST_CHECK(p == q);
  }
}

BOOST_AUTO_TEST_CASE(serialization) {
  auto p = Particle();

  auto const bond_id = 5;
  auto const bond_partners = std::array<const int, 3>{{12, 13, 14}};

  p.id() = 15;
  p.bonds().insert({bond_id, bond_partners});
  p.force() = {1., -2., 3.};
#ifdef ESPRESSO_ROTATION
  p.torque() = {-4., 5., -6.};
#endif
#ifdef ESPRESSO_EXCLUSIONS
  std::vector<int> el = {5, 6, 7, 8};
  p.exclusions() = Utils::compact_vector<int>{el.begin(), el.end()};
#endif

  std::stringstream stream;
  boost::archive::text_oarchive out_ar(stream);
  out_ar << p;

  boost::archive::text_iarchive in_ar(stream);
  auto q = Particle();
  in_ar >> q;

  auto const &pf = std::as_const(p).force_and_torque();
  BOOST_CHECK(q.id() == p.id());
  BOOST_CHECK((*q.bonds().begin() == BondView{bond_id, bond_partners}));
  BOOST_TEST(q.force() == pf.f, boost::test_tools::per_element());
#ifdef ESPRESSO_ROTATION
  BOOST_TEST(q.torque() == pf.torque, boost::test_tools::per_element());
#endif
  check_particle_force(q.force_and_torque(), pf);
}

namespace Utils {
template <>
struct is_statically_serializable<ParticleProperties> : std::true_type {};
} // namespace Utils

BOOST_AUTO_TEST_CASE(properties_serialization) {
  auto const expected_size =
      Utils::MemcpyOArchive::packing_size<ParticleProperties>();

  BOOST_CHECK_LE(expected_size, sizeof(ParticleProperties));

  std::vector<char> buf(expected_size);

  auto prop = ParticleProperties{};
  prop.identity = 1234;

  {
    auto oa = Utils::MemcpyOArchive{buf};

    oa << prop;

    BOOST_CHECK_EQUAL(oa.bytes_written(), expected_size);
  }

  {
    auto ia = Utils::MemcpyIArchive{buf};
    ParticleProperties out;

    ia >> out;
    BOOST_CHECK_EQUAL(ia.bytes_read(), expected_size);
    BOOST_CHECK_EQUAL(out.identity, prop.identity);
  }
}


#if defined(ESPRESSO_THERMAL_STONER_WOHLFARTH) || defined(ESPRESSO_EGG_MODEL) || defined(ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET)
namespace Utils {
template <>
struct is_statically_serializable<ParticleMagnetodynamicsParameters>
    : std::true_type {};
} // namespace Utils

BOOST_AUTO_TEST_CASE(magnetodynamics_properties_serialization) {
  auto const expected_size =
      Utils::MemcpyOArchive::packing_size<ParticleMagnetodynamicsParameters>();

  BOOST_CHECK_LE(expected_size, sizeof(ParticleMagnetodynamicsParameters));

  std::vector<char> buf(expected_size);

  auto magnetodynamics = ParticleMagnetodynamicsParameters{};
#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
  magnetodynamics.tsw.is_enabled = true;
  magnetodynamics.tsw.phi0 = 0.25;
  magnetodynamics.tsw.sat_mag = 1.5;
  magnetodynamics.tsw.ani_fld_inv = 0.75;
  magnetodynamics.tsw.ani_energy = 2.5;
  magnetodynamics.tsw.tau0_inv = 3.5;
  magnetodynamics.tsw.dt_incr = 4.5;
#endif
#ifdef ESPRESSO_EGG_MODEL
  magnetodynamics.egg.is_enabled = false;
  magnetodynamics.egg.gamma = 1.25;
  magnetodynamics.egg.anisotropy_energy = 6.5;
  magnetodynamics.egg.axis_quat_body = Utils::Quaternion<double>{{1., 2., 3., 4.}};
  magnetodynamics.egg.axis_quat_space = Utils::Quaternion<double>{{4., 3., 2., 1.}};
  magnetodynamics.egg.internal_magnetic_torque = {7., 8., 9.};
#endif
#ifdef ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET
  magnetodynamics.ideal.is_enabled = true;
  magnetodynamics.ideal.sat_mag = 5.5;
#endif

  {
    auto oa = Utils::MemcpyOArchive{buf};
    oa << magnetodynamics;
    BOOST_CHECK_EQUAL(oa.bytes_written(), expected_size);
  }

  {
    auto ia = Utils::MemcpyIArchive{buf};
    ParticleMagnetodynamicsParameters out;
    ia >> out;
    BOOST_CHECK_EQUAL(ia.bytes_read(), expected_size);
    BOOST_CHECK_EQUAL(out.enabled_count(), magnetodynamics.enabled_count());
#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH
    BOOST_CHECK_EQUAL(out.tsw.is_enabled, magnetodynamics.tsw.is_enabled);
    BOOST_CHECK_EQUAL(out.tsw.phi0, magnetodynamics.tsw.phi0);
    BOOST_CHECK_EQUAL(out.tsw.sat_mag, magnetodynamics.tsw.sat_mag);
    BOOST_CHECK_EQUAL(out.tsw.ani_fld_inv, magnetodynamics.tsw.ani_fld_inv);
    BOOST_CHECK_EQUAL(out.tsw.ani_energy, magnetodynamics.tsw.ani_energy);
    BOOST_CHECK_EQUAL(out.tsw.tau0_inv, magnetodynamics.tsw.tau0_inv);
    BOOST_CHECK_EQUAL(out.tsw.dt_incr, magnetodynamics.tsw.dt_incr);
#endif
#ifdef ESPRESSO_EGG_MODEL
    BOOST_CHECK_EQUAL(out.egg.is_enabled, magnetodynamics.egg.is_enabled);
    BOOST_CHECK_EQUAL(out.egg.gamma, magnetodynamics.egg.gamma);
    BOOST_CHECK_EQUAL(out.egg.anisotropy_energy, magnetodynamics.egg.anisotropy_energy);
    for (unsigned int i = 0; i < 4; ++i) {
      BOOST_CHECK_EQUAL(out.egg.axis_quat_body[i], magnetodynamics.egg.axis_quat_body[i]);
      BOOST_CHECK_EQUAL(out.egg.axis_quat_space[i], magnetodynamics.egg.axis_quat_space[i]);
    }
    BOOST_TEST(out.egg.internal_magnetic_torque == magnetodynamics.egg.internal_magnetic_torque,
               boost::test_tools::per_element());
#endif
#ifdef ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET
    BOOST_CHECK_EQUAL(out.ideal.is_enabled, magnetodynamics.ideal.is_enabled);
    BOOST_CHECK_EQUAL(out.ideal.sat_mag, magnetodynamics.ideal.sat_mag);
#endif
  }
}
#endif

namespace Utils {
template <>
struct is_statically_serializable<ParticleForce> : std::true_type {};
} // namespace Utils

BOOST_AUTO_TEST_CASE(force_serialization) {
  auto const expected_size =
      Utils::MemcpyOArchive::packing_size<ParticleForce>();

  BOOST_CHECK_LE(expected_size, sizeof(ParticleForce));

  std::vector<char> buf(expected_size);

  auto pf = ParticleForce{{1, 2, 3}};
#ifdef ESPRESSO_ROTATION
  pf.torque = {4, 5, 6};
#endif

  {
    auto oa = Utils::MemcpyOArchive{buf};

    oa << pf;

    BOOST_CHECK_EQUAL(oa.bytes_written(), expected_size);
  }

  {
    auto ia = Utils::MemcpyIArchive{buf};
    ParticleForce out;

    ia >> out;

    BOOST_CHECK_EQUAL(ia.bytes_read(), expected_size);
    check_particle_force(out, pf);
  }
}

BOOST_AUTO_TEST_CASE(force_constructors) {

  auto pf = ParticleForce{{1, 2, 3}};
#ifdef ESPRESSO_ROTATION
  pf.torque = {4, 5, 6};
#endif

  // check copy constructor
  {
    ParticleForce out(pf);
    check_particle_force(out, pf);
  }

  // check copy assignment operator
  {
    ParticleForce out; // avoid copy elision
    out = pf;
    check_particle_force(out, pf);
  }
}

#ifdef ESPRESSO_BOND_CONSTRAINT

void check_particle_rattle(ParticleRattle const &out,
                           ParticleRattle const &ref) {
  BOOST_TEST(out.correction == ref.correction,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(rattle_serialization) {
  auto const expected_size =
      Utils::MemcpyOArchive::packing_size<ParticleRattle>();

  BOOST_CHECK_LE(expected_size, sizeof(ParticleRattle));

  std::vector<char> buf(expected_size);

  auto pr = ParticleRattle{{1, 2, 3}};

  {
    auto oa = Utils::MemcpyOArchive{buf};

    oa << pr;

    BOOST_CHECK_EQUAL(oa.bytes_written(), expected_size);
  }

  {
    auto ia = Utils::MemcpyIArchive{buf};
    ParticleRattle out;

    ia >> out;

    BOOST_CHECK_EQUAL(ia.bytes_read(), expected_size);
    check_particle_rattle(out, pr);
  }
}

BOOST_AUTO_TEST_CASE(rattle_constructors) {
  auto pr = ParticleRattle{{1, 2, 3}};

  // check copy constructor
  {
    ParticleRattle out(pr);
    check_particle_rattle(out, pr);
  }

  // check copy assignment operator
  {
    ParticleRattle out; // avoid copy elision
    out = pr;
    check_particle_rattle(out, pr);
  }
}
#endif // ESPRESSO_BOND_CONSTRAINT

#ifdef ESPRESSO_THERMAL_STONER_WOHLFARTH

void check_particle_tsw(ThermalStonerWohlfarthParameters const &out,
                        ThermalStonerWohlfarthParameters const &ref) {
  BOOST_TEST(out.is_enabled == ref.is_enabled);
  BOOST_TEST(out.phi0 == ref.phi0);
  BOOST_TEST(out.sat_mag == ref.sat_mag);
  BOOST_TEST(out.ani_fld_inv == ref.ani_fld_inv);
  BOOST_TEST(out.ani_energy == ref.ani_energy);
  BOOST_TEST(out.tau0_inv == ref.tau0_inv);
  BOOST_TEST(out.dt_incr == ref.dt_incr);
}

BOOST_AUTO_TEST_CASE(thermal_stoner_wohlfarth_serialization) {
  auto const expected_size =
      Utils::MemcpyOArchive::packing_size<ThermalStonerWohlfarthParameters>();

  BOOST_CHECK_LE(expected_size, sizeof(ThermalStonerWohlfarthParameters));

  std::vector<char> buf(expected_size);

  auto pr = ThermalStonerWohlfarthParameters{.is_enabled = false,
                                             .phi0 = 1.,
                                             .sat_mag = 2.,
                                             .ani_fld_inv = 3.,
                                             .ani_energy = 4.,
                                             .tau0_inv = 5.,
                                             .dt_incr = 6.};

  {
    auto oa = Utils::MemcpyOArchive{buf};

    oa << pr;

    BOOST_CHECK_EQUAL(oa.bytes_written(), expected_size);
  }

  {
    auto ia = Utils::MemcpyIArchive{buf};
    ThermalStonerWohlfarthParameters out;

    ia >> out;

    BOOST_CHECK_EQUAL(ia.bytes_read(), expected_size);
    check_particle_tsw(out, pr);
  }
}

BOOST_AUTO_TEST_CASE(thermal_stoner_wohlfarth_constructors) {
  auto pr = ThermalStonerWohlfarthParameters{.is_enabled = false,
                                             .phi0 = 1.,
                                             .sat_mag = 2.,
                                             .ani_fld_inv = 3.,
                                             .ani_energy = 4.,
                                             .tau0_inv = 5.,
                                             .dt_incr = 6.};

  // check copy constructor
  {
    ThermalStonerWohlfarthParameters out(pr);
    check_particle_tsw(out, pr);
  }

  // check copy assignment operator
  {
    ThermalStonerWohlfarthParameters out; // avoid copy elision
    out = pr;
    check_particle_tsw(out, pr);
  }
}
#endif // ESPRESSO_THERMAL_STONER_WOHLFARTH

#ifdef ESPRESSO_EGG_MODEL

void check_particle_egg_model(EggModelParameters const &out,
                              EggModelParameters const &ref) {
  BOOST_TEST(out.is_enabled == ref.is_enabled);
  BOOST_TEST(out.gamma == ref.gamma);
  BOOST_TEST(out.anisotropy_energy == ref.anisotropy_energy);
  for (unsigned int i = 0; i < 4; i++) {
    BOOST_TEST(out.axis_quat_body[i] == ref.axis_quat_body[i]);
    BOOST_TEST(out.axis_quat_space[i] == ref.axis_quat_space[i]);
  }
  BOOST_TEST(out.internal_magnetic_torque == ref.internal_magnetic_torque,
             boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(egg_model_serialization) {
  auto const expected_size =
      Utils::MemcpyOArchive::packing_size<EggModelParameters>();

  BOOST_CHECK_LE(expected_size, sizeof(EggModelParameters));

  std::vector<char> buf(expected_size);

  auto pr = EggModelParameters{};
  pr.is_enabled = true;
  pr.gamma = 2.;
  pr.anisotropy_energy = 3.;
  pr.axis_quat_body = {4., 5., 6., 7.};
  pr.axis_quat_space = {8., 9., 10., 11.};
  pr.internal_magnetic_torque = {12., 13., 14.};

  {
    auto oa = Utils::MemcpyOArchive{buf};

    oa << pr;

    BOOST_CHECK_EQUAL(oa.bytes_written(), expected_size);
  }

  {
    auto ia = Utils::MemcpyIArchive{buf};
    EggModelParameters out;

    ia >> out;

    BOOST_CHECK_EQUAL(ia.bytes_read(), expected_size);
    check_particle_egg_model(out, pr);
  }
}

BOOST_AUTO_TEST_CASE(egg_model_constructors) {
  auto pr = EggModelParameters{};
  pr.is_enabled = true;
  pr.gamma = 2.;
  pr.anisotropy_energy = 3.;
  pr.axis_quat_body = {4., 5., 6., 7.};
  pr.axis_quat_space = {8., 9., 10., 11.};
  pr.internal_magnetic_torque = {12., 13., 14.};

  {
    EggModelParameters out(pr);
    check_particle_egg_model(out, pr);
  }

  {
    EggModelParameters out;
    out = pr;
    check_particle_egg_model(out, pr);
  }
}
#endif // ESPRESSO_EGG_MODEL

#ifdef ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET

void check_particle_ideal_magnetizable_superparamagnet(IdealMagnetizableSuperparamagnetParameters const &out,
                              IdealMagnetizableSuperparamagnetParameters const &ref) {
  BOOST_TEST(out.is_enabled == ref.is_enabled);
  BOOST_TEST(out.sat_mag == ref.sat_mag);
}

BOOST_AUTO_TEST_CASE(ideal_magnetizable_superparamagnet_serialization) {
  auto const expected_size =
      Utils::MemcpyOArchive::packing_size<IdealMagnetizableSuperparamagnetParameters>();

  BOOST_CHECK_LE(expected_size, sizeof(IdealMagnetizableSuperparamagnetParameters));

  std::vector<char> buf(expected_size);

  auto pr = IdealMagnetizableSuperparamagnetParameters{.is_enabled = true, .sat_mag = 2.};

  {
    auto oa = Utils::MemcpyOArchive{buf};

    oa << pr;

    BOOST_CHECK_EQUAL(oa.bytes_written(), expected_size);
  }

  {
    auto ia = Utils::MemcpyIArchive{buf};
    IdealMagnetizableSuperparamagnetParameters out;

    ia >> out;

    BOOST_CHECK_EQUAL(ia.bytes_read(), expected_size);
    check_particle_ideal_magnetizable_superparamagnet(out, pr);
  }
}

BOOST_AUTO_TEST_CASE(ideal_magnetizable_superparamagnet_constructors) {
  auto pr = IdealMagnetizableSuperparamagnetParameters{.is_enabled = true, .sat_mag = 2.};

  {
    IdealMagnetizableSuperparamagnetParameters out(pr);
    check_particle_ideal_magnetizable_superparamagnet(out, pr);
  }

  {
    IdealMagnetizableSuperparamagnetParameters out;
    out = pr;
    check_particle_ideal_magnetizable_superparamagnet(out, pr);
  }
}
#endif // ESPRESSO_IDEAL_MAGNETIZABLE_SUPERPARAMAGNET

BOOST_AUTO_TEST_CASE(particle_bitfields) {
  auto p = Particle();

  // check default values
  BOOST_CHECK(not p.has_fixed_coordinates());
  BOOST_CHECK(not p.can_rotate());
  BOOST_CHECK(not p.is_fixed_along(1));
  BOOST_CHECK(not p.can_rotate_around(1));

  // check setting of one axis
#ifdef ESPRESSO_EXTERNAL_FORCES
  p.set_fixed_along(1, true);
  BOOST_CHECK(p.is_fixed_along(1));
  BOOST_CHECK(p.has_fixed_coordinates());
#endif
#ifdef ESPRESSO_ROTATION
  p.set_can_rotate_around(1, true);
  BOOST_CHECK(p.can_rotate_around(1));
  BOOST_CHECK(p.can_rotate());
#endif

  // check that unsetting is properly registered
#ifdef ESPRESSO_EXTERNAL_FORCES
  p.set_fixed_along(1, false);
  BOOST_CHECK(not p.has_fixed_coordinates());
#endif
#ifdef ESPRESSO_ROTATION
  p.set_can_rotate_around(1, false);
  BOOST_CHECK(not p.can_rotate());
#endif

  // check setting of all flags at once
#ifdef ESPRESSO_ROTATION
  p.set_can_rotate_all_axes();
  BOOST_CHECK(p.can_rotate_around(0));
  BOOST_CHECK(p.can_rotate_around(1));
  BOOST_CHECK(p.can_rotate_around(2));
  p.set_cannot_rotate_all_axes();
  BOOST_CHECK(not p.can_rotate());
#endif

  static_assert(
      std::is_same_v<std::underlying_type_t<PropagationMode::PropagationMode>,
                     decltype(ParticleProperties::propagation)>);
}
