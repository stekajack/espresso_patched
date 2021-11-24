# Copyright (C) 2010-2019 The ESPResSo project
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

import espressomd
import espressomd.shapes
import unittest as ut
import unittest_decorators as utx
import numpy as np

import tests_common

# Define the LB Parameters
TIME_STEP = 0.01
AGRID = 0.5
KVISC = 4
DENS = 2
G = 0.08
BOX_SIZE = 18 * AGRID

LB_PARAMS = {'agrid': AGRID,
             'density': DENS,
             'viscosity': KVISC,
             'tau': TIME_STEP,
             'ext_force_density': [0, DENS * G, 0]}
# System setup
RADIUS = 6 * AGRID


class Buoyancy(object):
    """
    Tests buoyancy force on a sphere in a closed box of lb fluid and
    the overall force balance

    """
    lbf = None
    system = espressomd.System(box_l=[BOX_SIZE] * 3)
    system.time_step = TIME_STEP
    system.cell_system.skin = 0.01

    def test(self):
        self.system.actors.clear()
        self.system.actors.add(self.lbf)
        self.lbf.clear_boundaries()

        # Setup walls
        for i in range(3):
            n = np.zeros(3)
            n[i] = 1
            self.lbf.add_boundary_from_shape(espressomd.shapes.Wall(
                normal=-n, dist=-(self.system.box_l[i] - AGRID)))
            self.lbf.add_boundary_from_shape(
                espressomd.shapes.Wall(normal=n, dist=AGRID))

        # setup sphere without slip in the middle
        sphere_shape = espressomd.shapes.Sphere(
            radius=RADIUS, center=self.system.box_l / 2, direction=1)
        self.lbf.add_boundary_from_shape(sphere_shape)

        sphere_volume = 4. / 3. * np.pi * RADIUS**3

        # Equilibration
        last_force = np.inf * np.ones(3)
        self.system.integrator.run(100)
        # TODO: WALBERLA: (#4381) boundary forces not reliable at the moment
        # while True:
        #     self.system.integrator.run(10)
        #     force = np.linalg.norm(sphere.get_force())
        #     if np.linalg.norm(force - last_force) < 0.01:
        #         break
        #     last_force = force

        # # Check force balance
        # boundary_force = np.zeros(3)
        # for b in self.system.lbboundaries:
        #     boundary_force += b.get_force()

        # fluid_nodes = tests_common.count_fluid_nodes(self.lbf)
        # fluid_volume = fluid_nodes * AGRID**3
        # applied_force = fluid_volume * np.array(LB_PARAMS['ext_force_density'])

        # np.testing.assert_allclose(
        #     boundary_force,
        #     applied_force,
        #     atol=0.08 * np.linalg.norm(applied_force))

        # # Check buoyancy force on the sphere
        # expected_force = np.array(
        #     [0, -sphere_volume * DENS * G, 0])
        # np.testing.assert_allclose(
        #     np.copy(sphere.get_force()), expected_force,
        #     atol=np.linalg.norm(expected_force) * 0.02)


@utx.skipIfMissingFeatures(["EXTERNAL_FORCES", "LB_WALBERLA"])
class LBCPUBuoyancy(ut.TestCase, Buoyancy):

    def setUp(self):
        self.lbf = espressomd.lb.LBFluidWalberla(**LB_PARAMS)


if __name__ == "__main__":
    ut.main()
