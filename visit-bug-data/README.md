# Kitchen time-varying test data (VisIt bug)

Files for reproducing the `time(mesh)` + singleton crash with a `.visit` metafile.

## On GitHub

Pushed to branch `visit-bug-data` in `dashkim/ComputerVisualizationProject`:

https://github.com/dashkim/ComputerVisualizationProject/tree/visit-bug-data/visit-bug-data

## Clone on your Mac

```bash
git clone -b visit-bug-data git@github.com:dashkim/ComputerVisualizationProject.git
cd ComputerVisualizationProject/visit-bug-data
```

Or sparse checkout of just this folder.

## Reproduce

```python
OpenDatabase("kitchen.visit")
DefineScalarExpression("foo", "time(mesh)")
AddPlot("Pseudocolor", "foo")
DrawPlots()
TimeSliderNextState()
```
