#!/usr/bin/env python3
"""VTK viewer: pick marching-cubes case 0–255; geometry from triCase rows in tricase.cxx."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Paul Bourke / Lorensen–Cline vertex order (matches triCase edge indices in this course).
# v0..v3 bottom face z=0; v4..v7 top z=1.
_CORNER_POS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]

# Edge i connects EDGE_VERT[i][0] -> EDGE_VERT[i][1]
_EDGE_VERT: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

_ROW_RE = re.compile(
    r"^\s*\{([^}]*)\}\s*,?\s*/\*\s*(\d+)\b(?:\s+[^*]*)?\*/",
    re.MULTILINE,
)


def parse_tricase(text: str) -> list[tuple[int, ...]]:
    """Parse static int triCase[256][16] body (same row pattern as merge_tricase.py)."""
    rows: dict[int, tuple[int, ...]] = {}
    for m in _ROW_RE.finditer(text):
        case = int(m.group(2))
        nums = tuple(int(x.strip()) for x in m.group(1).split(","))
        if len(nums) != 16:
            raise ValueError(f"case {case}: expected 16 ints, got {len(nums)}")
        rows[case] = nums
    if len(rows) != 256:
        raise ValueError(f"expected 256 rows, got {len(rows)}")
    return [rows[i] for i in range(256)]


def _edge_surface_point(
    edge_idx: int,
    inside: list[float],
) -> tuple[float, float, float] | None:
    if edge_idx < 0 or edge_idx >= 12:
        return None
    a, b = _EDGE_VERT[edge_idx]
    sa, sb = inside[a], inside[b]
    if abs(sa - sb) < 1e-9:
        return None
    u = (0.5 - sa) / (sb - sa)
    pa = _CORNER_POS[a]
    pb = _CORNER_POS[b]
    return (
        pa[0] + u * (pb[0] - pa[0]),
        pa[1] + u * (pb[1] - pa[1]),
        pa[2] + u * (pb[2] - pa[2]),
    )


def _triangles_from_row(row: tuple[int, ...]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    i = 0
    while i + 2 < len(row):
        e0, e1, e2 = row[i], row[i + 1], row[i + 2]
        if e0 < 0 or e1 < 0 or e2 < 0:
            break
        out.append((e0, e1, e2))
        i += 3
    return out


def _inside_scalars(case_index: int) -> list[float]:
    return [1.0 if (case_index >> i) & 1 else 0.0 for i in range(8)]


def _clear_headless_env_before_vtk_import() -> None:
    """VTK reads this at import time; unsetting it avoids a silent/no-GUI build in many setups."""
    key = "VTK_DEFAULT_RENDER_WINDOW_HEADLESS"
    if key not in os.environ:
        return
    val = os.environ.get(key, "")
    low = str(val).strip().lower()
    if low in ("", "0", "false", "no", "off"):
        return
    del os.environ[key]
    print(f"note: unset {key} so a normal window can open.", flush=True)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    default_cxx = repo / "tricase.cxx"

    ap = argparse.ArgumentParser(description="VTK marching-cube case viewer (reads triCase from tricase.cxx).")
    ap.add_argument(
        "--tricase",
        type=Path,
        default=default_cxx,
        help=f"path to tricase.cxx (default: {default_cxx})",
    )
    args = ap.parse_args()
    cxx_path: Path = args.tricase.resolve()
    if not cxx_path.is_file():
        print(f"error: file not found: {cxx_path}", file=sys.stderr)
        sys.exit(1)

    try:
        tri_case = parse_tricase(cxx_path.read_text())
    except ValueError as e:
        print(f"error parsing {cxx_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded triCase from: {cxx_path}", flush=True)

    _clear_headless_env_before_vtk_import()

    try:
        from vtkmodules.vtkCommonColor import vtkNamedColors
        from vtkmodules.vtkCommonCore import vtkFloatArray, vtkLookupTable, vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkTriangle, vtkVertex
        from vtkmodules.vtkFiltersCore import vtkGlyph3D
        from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkSphereSource
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
        from vtkmodules.vtkInteractionWidgets import (
            vtkSliderRepresentation2D,
            vtkSliderWidget,
        )
        from vtkmodules.vtkRenderingAnnotation import vtkCornerAnnotation
        from vtkmodules.vtkRenderingCore import (
            vtkActor,
            vtkPolyDataMapper,
            vtkRenderWindow,
            vtkRenderWindowInteractor,
            vtkRenderer,
        )
        # Required when importing vtk piecemeal: registers the real OpenGL/Cocoa window
        # so vtkRenderWindowInteractor::Start() actually runs the GUI event loop.
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    except ImportError as e:
        print("error: VTK Python bindings not installed. Try: pip install vtk", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    # --- build static actors (cube outline) ---
    named_colors = vtkNamedColors()
    cube = vtkCubeSource()
    cube.SetCenter(0.5, 0.5, 0.5)
    cube.SetXLength(1.0)
    cube.SetYLength(1.0)
    cube.SetZLength(1.0)
    cube_mapper = vtkPolyDataMapper()
    cube_mapper.SetInputConnection(cube.GetOutputPort())
    cube_actor = vtkActor()
    cube_actor.SetMapper(cube_mapper)
    cube_actor.GetProperty().SetRepresentationToWireframe()
    cube_actor.GetProperty().SetColor(named_colors.GetColor3d("Gray"))
    cube_actor.GetProperty().SetLineWidth(2.0)

    # Corner spheres (glyph)
    corner_pts = vtkPoints()
    corner_pts.SetNumberOfPoints(8)
    for i, p in enumerate(_CORNER_POS):
        corner_pts.SetPoint(i, p[0], p[1], p[2])
    corner_pd = vtkPolyData()
    corner_pd.SetPoints(corner_pts)
    corner_pd.SetVerts(vtkCellArray())
    for i in range(8):
        v = vtkVertex()
        v.GetPointIds().SetId(0, i)
        corner_pd.GetVerts().InsertNextCell(v)

    sphere = vtkSphereSource()
    sphere.SetRadius(0.055)
    sphere.SetPhiResolution(16)
    sphere.SetThetaResolution(16)
    glyph = vtkGlyph3D()
    glyph.SetInputData(corner_pd)
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.Update()

    corner_lut = vtkLookupTable()
    corner_lut.SetNumberOfTableValues(2)
    corner_lut.SetTableRange(0.0, 1.0)
    corner_lut.SetTableValue(0, 0.92, 0.25, 0.2, 1.0)  # outside
    corner_lut.SetTableValue(1, 0.15, 0.72, 0.28, 1.0)  # inside
    corner_lut.Build()

    corner_mapper = vtkPolyDataMapper()
    corner_mapper.SetInputConnection(glyph.GetOutputPort())
    corner_mapper.SetScalarModeToUsePointFieldData()
    corner_mapper.SetScalarRange(0.0, 1.0)
    corner_mapper.SetLookupTable(corner_lut)
    corner_mapper.SetUseLookupTableScalarRange(True)
    corner_actor = vtkActor()
    corner_actor.SetMapper(corner_mapper)

    # Mesh (updated per case)
    mesh_pd = vtkPolyData()
    mesh_mapper = vtkPolyDataMapper()
    mesh_mapper.SetInputData(mesh_pd)
    mesh_actor = vtkActor()
    mesh_actor.SetMapper(mesh_mapper)
    prop = mesh_actor.GetProperty()
    prop.SetColor(0.25, 0.55, 0.95)
    prop.SetOpacity(0.92)
    prop.EdgeVisibilityOn()
    prop.SetEdgeColor(0.05, 0.05, 0.05)
    prop.SetLineWidth(1.5)

    renderer = vtkRenderer()
    renderer.SetBackground(named_colors.GetColor3d("WhiteSmoke"))
    renderer.AddActor(cube_actor)
    renderer.AddActor(corner_actor)
    renderer.AddActor(mesh_actor)

    ren_win = vtkRenderWindow()
    ren_win.AddRenderer(renderer)
    ren_win.SetWindowName("Marching cube case viewer")
    ren_win.SetSize(960, 720)
    ren_win.SetPosition(80, 80)
    # Prefer an on-screen window (some environments default to offscreen).
    if hasattr(ren_win, "SetOffScreenRendering"):
        ren_win.SetOffScreenRendering(False)
    if hasattr(ren_win, "SetShowWindow"):
        ren_win.SetShowWindow(True)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(ren_win)
    style = vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    ann = vtkCornerAnnotation()
    ann.SetLinearFontScaleFactor(2)
    ann.SetNonlinearFontScaleFactor(1)
    ann.SetMaximumFontSize(18)
    ann.SetText(0, "")  # lower left
    renderer.AddViewProp(ann)

    state: dict = {
        "case": 1,
        "tri_case": tri_case,
        "cxx_path": str(cxx_path),
        "play": False,
        "timer_id": None,
        "type_case_buffer": "",
        "corner_pd": corner_pd,
        "mesh_pd": mesh_pd,
        "corner_mapper": corner_mapper,
        "mesh_mapper": mesh_mapper,
        "ann": ann,
        "ren_win": ren_win,
    }

    def update_ann() -> None:
        """HUD only (no geometry rebuild)."""
        case_index = state["case"]
        buf = state.get("type_case_buffer", "")
        bits = format(case_index, "08b")
        row = tri_case[case_index]
        row_preview = ", ".join(str(x) for x in row[:16])
        if buf:
            typing = f"\nType case → [{buf}]   Enter=apply  Esc=cancel  BackSpace"
        else:
            typing = "\nType digits 0–255, then Enter to jump to a case."
        ann.SetText(
            0,
            f"{cxx_path.name}  |  case {case_index}  (0b{bits})\n"
            f"row: {row_preview}{typing}\n"
            "Keys: n/p step  Home/End 0/255  space play/pause  q quit",
        )

    def sync_slider(case_index: int) -> None:
        rep = vtkSliderRepresentation2D.SafeDownCast(slider_widget.GetRepresentation())
        if rep is not None:
            rep.SetValue(float(case_index))

    def apply_case(case_index: int) -> None:
        case_index = max(0, min(255, int(case_index)))
        state["case"] = case_index
        state["type_case_buffer"] = ""
        inside = _inside_scalars(case_index)
        row = tri_case[case_index]

        # Corner scalars on glyph input (point id 0..7)
        arr = vtkFloatArray()
        arr.SetName("inside")
        arr.SetNumberOfTuples(8)
        for i in range(8):
            arr.SetValue(i, inside[i])
        corner_pd.GetPointData().SetScalars(arr)
        corner_pd.Modified()
        glyph.Update()
        corner_mapper.Update()

        # Mesh triangles
        pts = vtkPoints()
        cells = vtkCellArray()
        pid = 0
        for e0, e1, e2 in _triangles_from_row(row):
            p0 = _edge_surface_point(e0, inside)
            p1 = _edge_surface_point(e1, inside)
            p2 = _edge_surface_point(e2, inside)
            if p0 is None or p1 is None or p2 is None:
                continue
            i0 = pid
            pts.InsertNextPoint(p0)
            pid += 1
            i1 = pid
            pts.InsertNextPoint(p1)
            pid += 1
            i2 = pid
            pts.InsertNextPoint(p2)
            pid += 1
            tri = vtkTriangle()
            tri.GetPointIds().SetId(0, i0)
            tri.GetPointIds().SetId(1, i1)
            tri.GetPointIds().SetId(2, i2)
            cells.InsertNextCell(tri)

        mesh_pd.SetPoints(pts)
        mesh_pd.SetPolys(cells)
        mesh_pd.Modified()
        mesh_mapper.Update()

        update_ann()
        ren_win.Render()

    def on_slider(widget, _event) -> None:
        rep = vtkSliderRepresentation2D.SafeDownCast(widget.GetRepresentation())
        if rep is None:
            return
        apply_case(int(rep.GetValue()))
        sync_slider(state["case"])

    def on_timer(_obj, _event) -> None:
        if not state["play"]:
            return
        nxt = (state["case"] + 1) % 256
        apply_case(nxt)
        sync_slider(nxt)

    _kp_digits = {
        "KP_0": "0",
        "KP_1": "1",
        "KP_2": "2",
        "KP_3": "3",
        "KP_4": "4",
        "KP_5": "5",
        "KP_6": "6",
        "KP_7": "7",
        "KP_8": "8",
        "KP_9": "9",
    }

    def _digit_from_keysym(key: str) -> str | None:
        if len(key) == 1 and key in "0123456789":
            return key
        return _kp_digits.get(key)

    def on_key(_obj, _event) -> None:
        key = interactor.GetKeySym()

        if key == "q":
            if state.get("timer_id") is not None:
                interactor.DestroyTimer(state["timer_id"])
            interactor.ExitCallback()
            return

        if key == "Escape":
            if state.get("type_case_buffer"):
                state["type_case_buffer"] = ""
                update_ann()
                ren_win.Render()
                return
            if state.get("timer_id") is not None:
                interactor.DestroyTimer(state["timer_id"])
            interactor.ExitCallback()
            return

        digit = _digit_from_keysym(key)
        if digit is not None:
            buf = state.get("type_case_buffer", "")
            if len(buf) < 3:
                state["type_case_buffer"] = buf + digit
                update_ann()
                ren_win.Render()
            return

        if key == "BackSpace":
            buf = state.get("type_case_buffer", "")
            if buf:
                state["type_case_buffer"] = buf[:-1]
                update_ann()
                ren_win.Render()
            return

        if key in ("Return", "Enter", "KP_Enter"):
            buf = state.get("type_case_buffer", "")
            if buf:
                try:
                    v = int(buf)
                except ValueError:
                    v = state["case"]
                apply_case(max(0, min(255, v)))
                sync_slider(state["case"])
                ren_win.Render()
            return

        c = state["case"]
        if key == "n" or key == "Right":
            c = (c + 1) % 256
        elif key == "p" or key == "Left":
            c = (c - 1) % 256
        elif key == "Home":
            c = 0
        elif key == "End":
            c = 255
        elif key == "space":
            if state["play"]:
                state["play"] = False
                if state.get("timer_id") is not None:
                    interactor.DestroyTimer(state["timer_id"])
                    state["timer_id"] = None
            else:
                state["play"] = True
                state["timer_id"] = interactor.CreateRepeatingTimer(350)
            return
        else:
            return
        apply_case(c)
        sync_slider(c)

    # Slider
    slider_rep = vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(0.0)
    slider_rep.SetMaximumValue(255.0)
    slider_rep.SetValue(1.0)
    slider_rep.SetTitleText("case")
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(0.02, 0.12)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(0.35, 0.12)
    slider_rep.SetSliderLength(0.02)
    slider_rep.SetSliderWidth(0.02)
    slider_rep.SetTubeWidth(0.006)
    slider_rep.SetEndCapLength(0.01)
    slider_rep.SetEndCapWidth(0.02)
    slider_rep.GetSliderProperty().SetColor(0.2, 0.2, 0.6)
    slider_rep.GetTitleProperty().SetColor(0, 0, 0)
    slider_rep.GetLabelProperty().SetColor(0, 0, 0)
    slider_rep.GetTubeProperty().SetColor(0.5, 0.5, 0.5)
    slider_rep.GetCapProperty().SetColor(0.5, 0.5, 0.5)

    slider_widget = vtkSliderWidget()
    slider_widget.SetInteractor(interactor)
    slider_widget.SetRepresentation(slider_rep)
    slider_widget.SetAnimationModeToAnimate()
    slider_widget.EnabledOn()
    slider_widget.AddObserver("InteractionEvent", on_slider)

    interactor.AddObserver("KeyPressEvent", on_key)
    interactor.AddObserver("TimerEvent", on_timer)

    try:
        interactor.Initialize()
    except Exception as e:
        print(f"error: VTK interactor Initialize() failed: {e}", file=sys.stderr)
        sys.exit(1)

    apply_case(1)
    ren_win.Render()
    interactor.ProcessEvents()
    if hasattr(ren_win, "WaitForCompletion"):
        ren_win.WaitForCompletion()
    if hasattr(ren_win, "RaiseWindow"):
        ren_win.RaiseWindow()

    off = ren_win.GetOffScreenRendering() if hasattr(ren_win, "GetOffScreenRendering") else None
    if off:
        print(
            "warning: VTK reports offscreen rendering; you may not see a window. "
            "Try: conda install -c conda-forge vtk   or   python3.12 -m pip install vtk",
            file=sys.stderr,
            flush=True,
        )

    print(
        "Viewer running — use the 3D window (Dock / Cmd+Tab if hidden). "
        f"Python: {sys.executable}  |  VTK: {ren_win.GetClassName()}  |  q or close window to quit.",
        flush=True,
    )

    try:
        t0 = time.perf_counter()
        interactor.Start()
        elapsed = time.perf_counter() - t0
    except Exception as e:
        print(f"error: VTK Start() failed: {e}", file=sys.stderr)
        sys.exit(1)

    if elapsed < 0.25:
        print(
            "warning: the VTK window closed almost immediately. "
            "If that was not intentional, reinstall VTK (e.g. conda install -c conda-forge vtk) "
            "or try python3.12.",
            file=sys.stderr,
            flush=True,
        )

    print("Viewer closed.", flush=True)


if __name__ == "__main__":
    main()
