# Changelog
In the version number *X.Y.Z*, Z is incremented whenever tests pass. *Y* is
incremented for each new public merge. *X* is incremented for changes, which
will make old configs or model files work no more.

## 0.0.4

## 0.0.3
- Fixed bug in {any}`sinusoidal_gen._ppf`, which lead to deviation between
  statistical and stochastic simulation.
  
- {any}`test_quantNoiseFormula` now exports more random behaviors to draw
  operands from.
  
- {any}`test_quantNoiseFormula` now also derives an expected SQNR
  bitwidth-independent offset.
  
- {any}`test_quantNoiseFormula` asserts the simulated SQNR now for all
  simulated random processes.

## 0.0.2
- Added new random process {any}`sinusoidal_gen` to draw sinusoidal operands
  from. {any}`generateSimulationOperands` is prepared to use them.
  
- New test {any}`test_quantNoiseFormula` can now re-compute the famous
  {math}`1.76 + 6.02 n` equation.
  
- Fullscale signals are now defined by {any}`fullscale_gen` and
  {any}`generateSimulationOperands` uses that to generate fullscale
  signals without a nasty self-defined *CDF*.

## 0.0.1
Base version.
