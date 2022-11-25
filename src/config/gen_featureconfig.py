#
# Copyright (C) 2013-2022 The ESPResSo project
# Copyright (C) 2012 Olaf Lenz
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

import string
import inspect
import sys
import os
# find featuredefs.py
moduledir = os.path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(os.path.join(moduledir, '..'))
import featuredefs

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} DEFFILE HPPFILE CPPFILE", file=sys.stderr)
    exit(2)

path_def, path_hpp, path_cpp = sys.argv[1:5]

print(f"Reading definitions from {path_def}")
defs = featuredefs.defs(path_def)

disclaimer = f"""/*
 * WARNING: This file was generated automatically.
 * Do not modify it or your changes will be overwritten!
 * Modify features.def instead.
 */
"""

print(f"Writing {path_hpp}")
hfile = open(path_hpp, 'w')
hfile.write(disclaimer)
hfile.write("""
#ifndef ESPRESSO_SRC_CONFIG_CONFIG_FEATURES_HPP
#define ESPRESSO_SRC_CONFIG_CONFIG_FEATURES_HPP
""")

hfile.write("""
/*********************************/
/* Handle definitions from CMake */
/*********************************/

#include "config/cmake_config.hpp"

// rename external features
""")

external_template = string.Template("""
#if defined(ESPRESSO_BUILD_WITH_$feature)
#undef ESPRESSO_BUILD_WITH_$feature
#define $feature
#endif
""")
for feature in sorted(defs.externals):
    hfile.write(external_template.substitute(feature=feature))

hfile.write("""
/************************************/
/* Handle definitions from MyConfig */
/************************************/

#include "config/myconfig-final.hpp"

""")

# handle implications
hfile.write("""\
/***********************/
/* Handle implications */
/***********************/
""")
implication_template = string.Template("""
// $feature implies $implied
#if defined($feature) && !defined($implied)
#define $implied
#endif
""")
for feature, implied in sorted(defs.implications):
    hfile.write(implication_template.substitute(
        feature=feature, implied=implied))

hfile.write("\n")

# output warnings if internal features are set manually
hfile.write("""\
/*****************************************************/
/* Warn when derived switches are specified manually */
/*****************************************************/
""")
derivation_template = string.Template("""
// $feature equals $expr
#ifdef $feature
#warning $feature is a derived switch and should not be set manually!
#elif $cppexpr
#define $feature
#endif
""")
for feature, expr, cppexpr in sorted(defs.derivations):
    hfile.write(derivation_template.substitute(
        feature=feature, cppexpr=cppexpr, expr=expr))

hfile.write("""

extern char const *const FEATURES[];
extern char const *const FEATURES_ALL[];
extern unsigned int const NUM_FEATURES;
extern unsigned int const NUM_FEATURES_ALL;

#endif
""")

hfile.close()

print(f"Writing {path_cpp}")
cfile = open(path_cpp, "w")

cfile.write(disclaimer)
cfile.write(f"""
#include "config/config-features.hpp"
#include "config/config.hpp"

/***********************/
/* Handle requirements */
/***********************/
""")

requirement_string = """
// {feature} requires {expr}
#if defined({feature}) && !({cppexpr})
#error Feature {feature} requires {expr}
#endif
"""
for feature, expr, cppexpr in sorted(defs.requirements):
    cfile.write(
        requirement_string.format(
            feature=feature, cppexpr=cppexpr, expr=expr))

cfile.write("""
/****************/
/* Feature list */
/****************/

char const *const FEATURES[] = {
""")

feature_string = """
#ifdef {feature}
  "{feature}",
#endif
"""

for feature in sorted(defs.externals.union(defs.features, defs.derived)):
    cfile.write(feature_string.format(feature=feature))

cfile.write("""
};
unsigned int const NUM_FEATURES = sizeof(FEATURES) / sizeof(char*);
""")

cfile.write("""
/*********************/
/* Feature full list */
/*********************/

char const *const FEATURES_ALL[] = {\
""")

feature_string = """
  "{feature}","""

for feature in sorted(defs.allfeatures):
    cfile.write(feature_string.format(feature=feature))

cfile.write("""
};
unsigned int const NUM_FEATURES_ALL = sizeof(FEATURES_ALL) / sizeof(char*);
""")

cfile.close()
