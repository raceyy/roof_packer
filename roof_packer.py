"""
Rooftop PV module packing on a single roof plane, using Shapely.

=============================================================================
WHAT THIS DOES, IN ORDER
=============================================================================
  1. Shrink the roof polygon inward by the perimeter setback.      (_usable_area)
  2. Cut out the ridge setback strip, if a ridge line was given.    (_usable_area)
  3. Cut out every obstacle, each grown by its clearance.           (_usable_area)
       --> what survives is the "usable" region: an arbitrary shape,
           possibly with holes, possibly split into several pieces.
  4. Rotate that region so the slope runs along the Y axis.         (pack)
  5. Shrink the module's along-slope dimension by cos(pitch).       (pack)
  6. For each orientation and each grid phase offset:               (pack)
       a. lay a regular grid of rectangles over the BOUNDING BOX    (_candidate_grid)
       b. throw away every rectangle not fully inside the usable
          region -- this is where the real polygon shape, the
          obstacle holes and the setbacks actually bite
       c. remember the arrangement that placed the most modules
  7. Rotate the winning rectangles back to the original CRS.        (pack)

The bounding box in step 6a is scaffolding only. It decides WHERE TO TRY,
never what is accepted. Acceptance is `usable.contains(rect)` in step 6b.

=============================================================================
COORDINATE CONVENTION
=============================================================================
All input geometry is PLAN VIEW (straight off nadir aerial imagery), in
metres, in a projected CRS (e.g. GDA2020 / MGA zone 50, EPSG:7850 for Perth).

Azimuth is the compass bearing of the DOWN-SLOPE direction, degrees clockwise
from true north. In Australia a north-facing roof has azimuth ~0, and that is
the GOOD aspect. Pitch is degrees from horizontal.
"""

from dataclasses import dataclass, field
from math import cos, radians
from typing import List, Optional, Tuple

from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union


@dataclass
class Module:
    """Physical module dimensions in metres. Defaults ~ a 440 W panel."""
    length: float = 1.722   # long edge
    width: float = 1.134    # short edge
    watts: int = 440


@dataclass
class Constraints:
    """Clearances in metres. Verify against AS/NZS 5033 and the current CEC
    Installation Guidelines for your jurisdiction before relying on these."""
    perimeter_setback: float = 0.20   # off gutters, verges, hips
    ridge_setback: float = 0.30       # off the ridge line
    walkway_width: float = 0.60       # CEC-style access way (not yet applied)
    obstacle_clearance: float = 0.30  # around vents, whirlybirds, aircon
    inter_row_gap: float = 0.02       # rail spacing between rows
    inter_col_gap: float = 0.02


@dataclass
class Plane:
    polygon: Polygon          # plan-view outline, metres
    azimuth: float            # down-slope bearing, deg from north
    pitch: float              # deg from horizontal
    obstacles: List[Polygon] = field(default_factory=list)
    ridge_line: Optional[object] = None  # optional LineString


# ---------------------------------------------------------------------------
# STEPS 1-3: build the region modules are actually allowed to occupy.
# ---------------------------------------------------------------------------
def _usable_area(plane: Plane, c: Constraints) -> MultiPolygon:
    # STEP 1. A negative buffer erodes the polygon inward by a fixed distance,
    # correctly following every edge including the sloping hip edges. This is
    # the perimeter setback: no module may sit within this band of the roof edge.
    area = plane.polygon.buffer(-c.perimeter_setback)

    # A narrow roof plane can vanish entirely under erosion. Bail out early.
    if area.is_empty:
        return MultiPolygon()

    # STEP 2. Buffering a LineString produces a corridor either side of it.
    # Subtracting that corridor carves a keep-out strip along the ridge.
    if plane.ridge_line is not None and c.ridge_setback > 0:
        area = area.difference(plane.ridge_line.buffer(c.ridge_setback))

    # STEP 3. Grow each obstacle by its clearance, union them (so overlapping
    # clearances merge instead of being subtracted twice), then punch them out.
    # The result may now contain holes, or be split into disjoint pieces.
    if plane.obstacles:
        blocked = unary_union([o.buffer(c.obstacle_clearance) for o in plane.obstacles])
        area = area.difference(blocked)

    # Normalise the return type: difference() gives a Polygon or a MultiPolygon
    # depending on whether the region got split. Callers want one type.
    if isinstance(area, Polygon):
        area = MultiPolygon([area]) if not area.is_empty else MultiPolygon()
    return area


# ---------------------------------------------------------------------------
# STEP 6a: propose candidate positions. Deliberately dumb and fast.
# ---------------------------------------------------------------------------
def _candidate_grid(bounds: Tuple[float, float, float, float],
                    cell_w: float, cell_h: float,
                    gap_x: float, gap_y: float,
                    offset_x: float, offset_y: float) -> List[Polygon]:
    """Tile the bounding box with a regular grid of rectangles.

    These are PROPOSALS ONLY. Many will overhang the real roof edge or land on
    an obstacle; the caller filters them. Generating against the bbox rather
    than the polygon keeps this a simple double loop instead of a geometry
    traversal, and rejecting the bad ones later costs almost nothing.
    """
    minx, miny, maxx, maxy = bounds
    rects = []
    y = miny + offset_y
    while y + cell_h <= maxy:          # rows, walking up the slope
        x = minx + offset_x
        while x + cell_w <= maxx:      # columns, walking across the slope
            rects.append(box(x, y, x + cell_w, y + cell_h))
            x += cell_w + gap_x        # advance by module width plus rail gap
        y += cell_h + gap_y
    return rects


# ---------------------------------------------------------------------------
# STEPS 4-7: the search.
# ---------------------------------------------------------------------------
def pack(plane: Plane,
         module: Module = Module(),
         c: Constraints = Constraints(),
         offset_steps: int = 6) -> dict:
    """Pack modules onto one roof plane.

    Returns the module polygons (plan view, original CRS), the count, and the
    resulting DC capacity in kW.
    """

    usable = _usable_area(plane, c)
    if usable.is_empty:
        return {"modules": [], "count": 0, "kw": 0.0, "usable_area_m2": 0.0}

    # STEP 4. Rotate the usable region so the down-slope direction aligns with
    # the Y axis. This lets the grid generator stay axis-aligned -- rows run
    # across the slope, columns up it, exactly how modules are actually racked.
    # Shapely rotates counter-clockwise, compass bearings run clockwise, hence
    # rotating by +azimuth here and -azimuth at the end.
    rot = plane.azimuth
    work = affinity.rotate(usable, rot, origin="centroid", use_radians=False)

    # STEP 5. Foreshortening. Seen from directly above, a pitched roof
    # compresses along the slope by cos(pitch). A 1.722 m module on a 22.5 deg
    # roof spans only 1.591 m in plan view. Since we pack in plan-view
    # coordinates, the module's along-slope dimension must be shrunk to match,
    # or we will systematically UNDER-count modules on steep roofs.
    fore = cos(radians(plane.pitch))
    best = {"modules": [], "count": 0, "kw": 0.0}

    # Two racking orientations. After the step-4 rotation, X is across-slope
    # (unaffected by pitch) and Y is along-slope (foreshortened).
    #   portrait  = module long edge runs UP the slope   -> length gets shrunk
    #   landscape = module long edge runs ACROSS the slope -> width gets shrunk
    orientations = [
        ("portrait",  module.width,  module.length * fore),
        ("landscape", module.length, module.width  * fore),
    ]

    # STEP 6. Brute-force search over orientation x grid phase.
    #
    # Why phase matters: a grid pinned to the bbox corner is arbitrary. Nudging
    # the whole array half a module sideways can fit an extra column against an
    # angled hip edge. Sliding through offset_steps positions in each axis (36
    # combinations by default) costs milliseconds and routinely finds ~10% more
    # modules than a single fixed origin would.
    for name, cell_w, cell_h in orientations:
        for i in range(offset_steps):
            for j in range(offset_steps):
                # Offsets sweep across one full grid period in each axis.
                ox = (cell_w + c.inter_col_gap) * i / offset_steps
                oy = (cell_h + c.inter_row_gap) * j / offset_steps

                # 6a. Propose.
                cands = _candidate_grid(work.bounds, cell_w, cell_h,
                                        c.inter_col_gap, c.inter_row_gap, ox, oy)

                # 6b. THE ACTUAL TEST. contains() demands the rectangle lie
                # wholly inside the usable region. This is what enforces the
                # true roof shape, the setbacks and the obstacle holes -- a
                # module clipping a hip edge or overlapping a whirlybird's
                # clearance fails here and is discarded.
                placed = [r for r in cands if work.contains(r)]

                # 6c. Keep the best arrangement seen so far.
                if len(placed) > best["count"]:
                    best = {"modules": placed, "count": len(placed),
                            "orientation": name,
                            "kw": len(placed) * module.watts / 1000.0}

    # STEP 7. Undo the step-4 rotation so the output sits back in the input CRS,
    # ready to write to GeoJSON and overlay on the source imagery.
    # Note: rotate about the ORIGINAL centroid, matching the forward rotation.
    out = [affinity.rotate(r, -rot, origin=usable.centroid, use_radians=False)
           for r in best["modules"]]

    return {"modules": out,
            "count": best["count"],
            "kw": round(best["kw"], 2),
            "orientation": best.get("orientation"),
            "usable_area_m2": round(usable.area, 1)}


if __name__ == "__main__":
    # A 12 x 8 m north-facing plane at 22.5 deg, with a whirlybird and a vent.
    roof = Plane(
        polygon=box(0, 0, 12, 8),
        azimuth=0.0,     # faces north - the good aspect in Australia
        pitch=22.5,
        obstacles=[box(4.0, 5.0, 4.6, 5.6),   # whirlybird
                   box(8.5, 2.0, 9.3, 2.8)],  # vent pipe
    )

    # Sweeping pitch shows the foreshortening effect from step 5: the same
    # plan-view footprint holds more modules as the roof gets steeper, because
    # a steeper roof genuinely has more surface area under the same footprint.
    for pitch in (0, 15, 22.5, 30):
        roof.pitch = pitch
        r = pack(roof)
        print(f"pitch {pitch:>5}deg -> {r['count']:>3} modules "
              f"({r['kw']:>5} kW, {r['orientation']}), "
              f"usable {r['usable_area_m2']} m2")
