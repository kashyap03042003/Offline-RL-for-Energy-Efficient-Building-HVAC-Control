"""Download real TMYx weather files (EnergyPlus .epw) from climate.onebuilding.org.

Only the .epw member of each archive is extracted, by basename, into data/weather/<city>.epw."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))  # noqa: E702
import io
import os
import urllib.request
import zipfile

BASE = "https://climate.onebuilding.org/WMO_Region_4_North_and_Central_America/USA_United_States_of_America"
FILES = {
    "boston": "MA_Massachusetts/USA_MA_Boston-Logan.Intl.AP.725090_TMYx.2009-2023.zip",
    "phoenix": "AZ_Arizona/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMYx.2009-2023.zip",
    "chicago": "IL_Illinois/USA_IL_Chicago-CGS.Calumet.725344_TMYx.2009-2023.zip",
    "miami": "FL_Florida/USA_FL_Miami.Exec.AP.722029_TMYx.2009-2023.zip",
}
out_dir = pathlib.Path(__file__).resolve().parents[1] / "data" / "weather"
out_dir.mkdir(parents=True, exist_ok=True)
for city, rel in FILES.items():
    dest = out_dir / f"{city}.epw"
    if dest.exists():
        print("exists", dest); continue
    req = urllib.request.Request(f"{BASE}/{rel}", headers={"User-Agent": "hvac-orl-research/0.1"})
    data = urllib.request.urlopen(req, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        epw = [m for m in z.namelist() if m.lower().endswith(".epw")]
        if len(epw) != 1:
            raise RuntimeError(f"{rel}: expected one .epw, found {epw}")
        dest.write_bytes(z.read(epw[0]))   # written by our own name: no path traversal
    print("saved", dest, os.path.getsize(dest), "bytes")
