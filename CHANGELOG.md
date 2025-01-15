# Changelog
In the version number *X.Y.Z*, Z is incremented whenever tests pass. *Y* is
incremented for each new public merge. *X* is incremented for changes, which
will make old configs or model files work no more.

## 0.0.2
- Added new random process {any}`sinusoidal_gen` to draw sinusoidal operands
  from. {any}`generateSimulationOperands` is prepared to use them.
  
- New test {any}`test_quantNoiseFormula` can now re-compute the famous
  {math}`1.76 + 6.02 n` equation.

## 0.0.1
Base version.
