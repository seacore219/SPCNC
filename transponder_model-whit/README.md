# Transponder Model


- `src/tests/current_loop.py` : First attempt at a simplified model.
- `src/tests/pulse_model.py` : Pulse shape for Rnw.
- `doc/spcnc_transponder1.nb`: Mathematica notebook to simplify ODEs.
- `doc/main.tex`: latex write up the latest version can be found here: [transponder document](https://gitlab.com/spcnc/transponder_model/-/jobs/artifacts/main/raw/transponder_model.pdf?job=document)
- Python module starts at `src/spcnc`



## Model Implementation

<a href="https://gitlab.com/spcnc/transponder_model/-/jobs/artifacts/main/raw/results/current_loop.png?job=tests">
<img src="https://gitlab.com/spcnc/transponder_model/-/jobs/artifacts/main/raw/results/current_loop.png?job=tests" width="400px" />
</a>
<a href="https://gitlab.com/spcnc/transponder_model/-/jobs/artifacts/main/raw/results/pulse_model.png?job=tests">
<img src="https://gitlab.com/spcnc/transponder_model/-/jobs/artifacts/main/raw/results/pulse_model.png?job=tests" width="400px" />
</a>
<a href="https://gitlab.com/spcnc/transponder_model/-/jobs/artifacts/main/raw/results/output_pulse.png?job=tests">
<img src="https://gitlab.com/spcnc/transponder_model/-/jobs/artifacts/main/raw/results/output_pulse.png?job=tests" width="400px" />
</a>


## Todo:

 - [ ] Create input labeled input data (2 spikes input)  
 - [ ] Construct simple network layouts:
   - [ ] 1 transponder at each sensor. Sensor input pulse should be followed by a delay to transponder to model physical layout. t1 output connected to t2 input: t1->t2 (no recursive network)
   - [ ] Same as above but with recursive network connectivity (add t2->t1)
   - [ ] Add two nodes near each sensor A_{t1}, A_{t2} at sensor location A and B_{t1}, B_{t2} at sensor location B. Connect with/without local recursion and with/without global recursion. For example local recursion sensor A: A_{t1}->A_{t2} and A_{t2}->A_{t1} without global reccursion: A_{ti}->B_{ti}  (no B->A).
 - [ ] Define an output layer

### Transponder construction
- The nTron is modeled as a variable resistor.

TBD:
- Input is a simple LR  series 
- Output  L in series and R shut to ground 
- Connections: Need Lk as a function of wire length
  Estimated from "inductivity" in spice model: ~ 50 pH/sq  
  or ~ 5 nH/um connection.



