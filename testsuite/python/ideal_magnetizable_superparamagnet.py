#
# Copyright (C) 2025-2026 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import unittest as ut
import unittest_decorators as utx
import espressomd
import espressomd.magnetostatics
import espressomd.propagation
Propagation = espressomd.propagation.Propagation


@utx.skipIfMissingFeatures(["IDEAL_MAGNETIZABLE_SUPERPARAMAGNET", "EXTERNAL_FORCES"])
class Test(ut.TestCase):
    system = espressomd.System(box_l=(10., 10., 10.))
    default_magnetodynamics = {
        "is_enabled": True,
        "sat_mag": 1.732,
    }
    initial_dipm = 1e-3
    field = 0.8
    temperature = 1.0
    gamma_T = 1.0
    seed = 42

    def setUp(self):
        self.system.cell_system.skin = 0.4
        self.system.min_global_cut = 1.
        self.system.time_step = 0.001
        self.system.periodicity = [True, True, True]
        self.system.thermostat.set_langevin(
            kT=self.temperature, gamma=self.gamma_T, seed=self.seed)

    def tearDown(self):
        self.system.part.clear()
        self.system.thermostat.turn_off()
        self.system.magnetostatics.clear()
        for x in self.system.constraints:
            self.system.constraints.remove(x)

    @staticmethod
    def python_magnetize_reference(dip_magnitude, h_ext, dip_fld):
        h_tot = np.array(h_ext) + np.array(dip_fld)
        tri = np.linalg.norm(h_tot)
        if tri < 1e-5:
            return np.zeros(3)
        dip_tri = dip_magnitude * tri
        return (dip_magnitude / tri *
                (1.0 / np.tanh(dip_tri) - 1.0 / dip_tri) * h_tot)

    def _init_virtual_site_pair(self):
        self.system.part.clear()
        p1 = self.system.part.add(pos=[0, 0, 0], director=[1, 0, 0],
                                  rotation=[False, False, False],
                                  fix=[True, True, True])
        p2 = self.system.part.add(
            pos=p1.pos, dip=[1, 2, 3], rotation=[False, False, False])
        p2.magnetodynamics.ideal = self.default_magnetodynamics
        p2.vs_auto_relate_to(p1)
        p2.propagation = (Propagation.TRANS_VS_RELATIVE |
                          Propagation.ROT_VS_INDEPENDENT)
        return p1, p2

    def _init_two_seeded_virtual_dipoles(self):
        self.system.part.clear()
        p1 = self.system.part.add(pos=[4.45, 5.0, 5.0],
                                  rotation=[False, False, False],
                                  fix=[True, True, True])
        dip1 = np.array([1.0, 0.4, -0.2])
        dip1 *= self.initial_dipm / np.linalg.norm(dip1)
        p2 = self.system.part.add(pos=p1.pos, dip=dip1,
                                  rotation=[False, False, False])
        p2.vs_auto_relate_to(p1)
        p2.propagation = (Propagation.TRANS_VS_RELATIVE |
                          Propagation.ROT_VS_INDEPENDENT)

        p3 = self.system.part.add(pos=[5.55, 5.0, 5.0],
                                  rotation=[False, False, False],
                                  fix=[True, True, True])
        dip2 = np.array([-0.3, 0.8, 0.5])
        dip2 *= self.initial_dipm / np.linalg.norm(dip2)
        p4 = self.system.part.add(pos=p3.pos, dip=dip2,
                                  rotation=[False, False, False])
        p4.vs_auto_relate_to(p3)
        p4.propagation = (Propagation.TRANS_VS_RELATIVE |
                          Propagation.ROT_VS_INDEPENDENT)
        return p2, p4

    def test_magnetodynamics_property(self):
        _, p2 = self._init_virtual_site_pair()
        self.assertEqual(set(p2.magnetodynamics.ideal.keys()),
                         {"is_enabled", "sat_mag"})
        self.assertEqual(p2.magnetodynamics.ideal["is_enabled"], True)
        self.assertAlmostEqual(p2.magnetodynamics.ideal["sat_mag"], 1.732)

    @utx.skipIfMissingFeatures(["DIPOLE_FIELD_TRACKING", "DIPOLES"])
    def test_pair_dipole_field_matches_python_magnetize(self):
        for h_ext in [np.zeros(3), np.array([0.15, -0.2, 0.05])]:
            with self.subTest(h_ext=h_ext):
                for constraint in list(self.system.constraints):
                    self.system.constraints.remove(constraint)
                self.system.magnetostatics.clear()
                p2, p4 = self._init_two_seeded_virtual_dipoles()
                self.system.magnetostatics.solver = \
                    espressomd.magnetostatics.DipolarDirectSum(prefactor=1.0)
                self.system.integrator.run(0, recalc_forces=True)

                dip_flds = [np.copy(p2.dip_fld), np.copy(p4.dip_fld)]
                self.assertGreater(
                    min(np.linalg.norm(x) for x in dip_flds), 1e-5)

                ext_h = espressomd.constraints.HomogeneousMagneticField(H=h_ext)
                self.system.constraints.add(ext_h)
                p2.magnetodynamics.ideal = self.default_magnetodynamics
                p4.magnetodynamics.ideal = self.default_magnetodynamics
                self.system.integrator.run(1)

                for particle, dip_fld in zip([p2, p4], dip_flds):
                    expected = self.python_magnetize_reference(
                        self.default_magnetodynamics["sat_mag"],
                        h_ext, dip_fld)
                    expected_without_dip_fld = self.python_magnetize_reference(
                        self.default_magnetodynamics["sat_mag"],
                        h_ext, np.zeros(3))

                    self.assertGreater(
                        np.linalg.norm(expected - expected_without_dip_fld),
                        1e-7)
                    np.testing.assert_allclose(np.copy(particle.dip), expected,
                                               rtol=1e-12, atol=1e-12)

    def test_minimal_no_field(self):
        _, p2 = self._init_virtual_site_pair()
        self.system.integrator.run(1)
        np.testing.assert_allclose(np.copy(p2.dip), np.zeros(3), atol=1e-12)

    def test_minimal_field(self):
        _, p2 = self._init_virtual_site_pair()
        ext_h = espressomd.constraints.HomogeneousMagneticField(
            H=(0., 0., self.field))
        self.system.constraints.add(ext_h)
        self.system.integrator.run(1)

        alpha = (self.default_magnetodynamics["sat_mag"] * self.field /
                 self.temperature)
        expected_dipm = self.default_magnetodynamics["sat_mag"] * (
            1.0 / np.tanh(alpha) - 1.0 / alpha)

        np.testing.assert_allclose(np.copy(p2.director), np.array([0., 0., 1.]),
                                   atol=1e-12)
        np.testing.assert_allclose(np.copy(p2.dip),
                                   np.array([0., 0., expected_dipm]),
                                   atol=1e-12)


if __name__ == "__main__":
    ut.main()
