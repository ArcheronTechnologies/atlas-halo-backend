# District-Level Polygons Strategy

**Goal:** Increase polygon coverage from 16% to 80%+ by using larger administrative areas

---

## The Problem

**Current situation:**
- 638 predictions generated
- Only 105 (16%) have polygon boundaries
- Most show as circles (fallback)

**Why so low?**
- Police reports use colloquial names: "Hjulsta backar", "Stockholm centrum"
- OSM has granular property names: specific buildings, blocks
- Name matching fails for most areas

---

## The Solution: Multi-Level Matching

### Level 1: Specific Neighborhoods (Current)
- **Source:** `neighborhood_polygons.json` (5,354 polygons)
- **Coverage:** 16% match rate
- **Examples:** Värnhem, Rosengård, Södermalm
- **Precision:** Very high - exact neighborhood boundaries

### Level 2: Districts/Suburbs (NEW!)
- **Source:** `district_polygons.json` (fetching now...)
- **Types:**
  - `place=suburb` (official suburbs)
  - `place=quarter` (city quarters)
  - `admin_level=9` (district administrative areas)
  - `admin_level=10` (neighborhood administrative areas)
- **Expected coverage:** 60-80% match rate
- **Examples:** "Norrmalm" (district), "Södermalm" (suburb)
- **Precision:** Medium - larger areas, less specific

### Level 3: City Bounds (Fallback)
- **Source:** City-level bounding boxes
- **Coverage:** 100% (everything gets matched)
- **Precision:** Low - entire city

---

## Matching Strategy (Cascading)

```python
def find_polygon_for_prediction(neighborhood_name, city_name):
    # Try Level 1: Exact neighborhood match
    polygon = match_neighborhood(neighborhood_name, city_name)
    if polygon:
        return polygon, "neighborhood"

    # Try Level 2: District/suburb match
    polygon = match_district(neighborhood_name, city_name)
    if polygon:
        return polygon, "district"

    # Try Level 3: City bounds (very large)
    polygon = get_city_bounds(city_name)
    if polygon:
        return polygon, "city"

    # Fallback: Circle
    return None, "circle"
```

---

## Expected Results

### Before (Current):
```
Total predictions: 638
  ✅ Polygons: 105 (16%)
  ⭕ Circles: 533 (84%)
```

### After (Districts Added):
```
Total predictions: 638
  ✅ Neighborhood polygons: 105 (16%)  ← Very precise
  ✅ District polygons: 400 (63%)      ← Medium precision
  ⭕ City bounds: 100 (16%)            ← Low precision
  ⭕ Circles: 33 (5%)                  ← Rare edge cases
```

**Total with ANY polygon: 605/638 = 95% coverage!**

---

## Implementation Steps

### 1. Fetch District Polygons ✅ (Running now)
```bash
python3 backend/data/fetch_osm_districts.py
```

**Output:** `backend/constants/district_polygons.json`

### 2. Merge Polygon Sources
Create `backend/constants/merged_polygons.json`:
```json
{
  "Stockholm": {
    "Värnhem": {
      "geometry": {...},
      "level": "neighborhood",
      "precision": "high"
    },
    "Södermalm": {
      "geometry": {...},
      "level": "district",
      "precision": "medium"
    }
  }
}
```

### 3. Update Prediction Worker
Modify `generate_predictions.py` to:
1. Load merged polygon data
2. Try neighborhood match first
3. Fall back to district match
4. Include `precision` field in response

### 4. Update Mobile App (Optional)
Add visual indicator for precision:
- High precision (neighborhood): Solid border
- Medium precision (district): Dashed border
- Low precision (city): Dotted border

---

## Data Sources

### Neighborhood Polygons (Already Have):
- **File:** `neighborhood_polygons.json` (9.0 MB)
- **Count:** 5,354 polygons
- **Cities:** Stockholm (3,484), Malmö (689), Göteborg (233), etc.
- **Format:** GeoJSON Polygon with center point

### District Polygons (Fetching Now):
- **File:** `district_polygons.json` (TBD)
- **Count:** Estimated 200-500 districts
- **Cities:** Top 12 Swedish cities
- **Format:** GeoJSON Polygon with center point
- **Queries:**
  - `place=suburb` - Official suburbs
  - `place=quarter` - City quarters
  - `admin_level=9,10` - Administrative districts

---

## Benefits

### For Users:
- ✅ 95% of predictions show as polygons (not circles)
- ✅ More professional appearance
- ✅ Better geographic understanding
- ✅ Larger areas are appropriately represented

### For System:
- ✅ Graceful degradation (precise → less precise → fallback)
- ✅ No breaking changes
- ✅ Easy to add more data sources later
- ✅ Can track match quality with precision field

---

## Fallback Chain Example

**Police Report:** "Incident in Hjulsta backar area, Stockholm"

**Matching Process:**
1. Try "Hjulsta backar" in Stockholm neighborhoods → ❌ Not found
2. Try fuzzy match "Hjulsta" → ❌ Not in neighborhood data
3. Try "Hjulsta" in Stockholm districts → ✅ **Found!** (suburb-level)
4. Return Hjulsta district polygon (medium precision)

**Result:** User sees polygon for Hjulsta suburb (larger area, but still useful!)

---

## Status

- [x] Neighborhood polygons exist (5,354)
- [x] Created district fetch script
- [ ] Fetching district polygons (running now...)
- [ ] Merge polygon sources
- [ ] Update prediction worker
- [ ] Test coverage improvement

**Expected completion:** 10-15 minutes for fetch, 30 minutes total implementation
