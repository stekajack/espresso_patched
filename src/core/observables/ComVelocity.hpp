#ifndef OBSERVABLES_COMVELOCITY_HPP
#define OBSERVABLES_COMVELOCITY_HPP


#include "PidObservable.hpp"
#include "particle_data.hpp" 
#include <vector>
#include "integrate.hpp"  


namespace Observables {


class ComVelocity : public PidObservable {
public:
    virtual int n_values() const override { return 3; }
    virtual int actual_calculate() override {
  if (!sortPartCfg()) {
      runtimeErrorMsg() <<"could not sort partCfg";
    return -1;
  }
  double total_mass=0;
  for (int i = 0; i<ids.size(); i++ ) {
    if (ids[i] >= n_part)
      return 1;
    double mass = partCfg[ids[i]].p.mass;
    last_value[0] += mass * partCfg[ids[i]].m.v[0];
    last_value[1] += mass * partCfg[ids[i]].m.v[1];
    last_value[2] += mass * partCfg[ids[i]].m.v[2];
    total_mass+=mass;
  }
  last_value[0]/=total_mass*time_step;
  last_value[1]/=total_mass*time_step;
  last_value[2]/=total_mass*time_step;
  return 0;
};
};

} // Namespace Observables
#endif

