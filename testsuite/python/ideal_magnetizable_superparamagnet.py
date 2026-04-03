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
import espressomd.propagation
Propagation = espressomd.propagation.Propagation


@utx.skipIfMissingFeatures(["IDEAL_MAGNETIZABLE_SUPERPARAMAGNET", "EXTERNAL_FORCES"])
class Test(ut.TestCase):
    system = espressomd.System(box_l=(10., 10., 10.))
    default_magnetodynamics = {
        "is_enabled": True,
        "sat_mag": 2.0,
    }
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
        for x in self.system.constraints:
            self.system.constraints.remove(x)

    def _init_virtual_site_pair(self):
        self.system.part.clear()
        p1 = self.system.part.add(pos=[0, 0, 0], director=[1, 0, 0],
                                  rotation=[False, False, False],
                                  fix=[True, True, True])
        p2 = self.system.part.add(
            pos=p1.pos, dip=[1, 2, 3], rotation=[False, False, False],
            magnetodynamics=self.default_magnetodynamics)
        p2.vs_auto_relate_to(p1)
        p2.propagation = (Propagation.TRANS_VS_RELATIVE |
                          Propagation.ROT_VS_INDEPENDENT)
        return p1, p2

    def test_magnetodynamics_property(self):
        _, p2 = self._init_virtual_site_pair()
        self.assertEqual(set(p2.magnetodynamics.keys()),
                         {"is_enabled", "sat_mag"})
        self.assertEqual(p2.magnetodynamics["is_enabled"], True)
        self.assertAlmostEqual(p2.magnetodynamics["sat_mag"], 2.0)

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
