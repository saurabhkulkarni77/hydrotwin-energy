#!/usr/bin/env python3
"""Discover NRCS AWDB stations (SOAP) for the HydroTwin Energy regions.
Prints candidate SNOTEL/SCAN triplets with metadata so picks are auditable."""
import requests, re, sys
from xml.etree import ElementTree as ET

URL = "https://wcc.sc.egov.usda.gov/awdbWebService/services"
NS = "http://www.wcc.nrcs.usda.gov/ns/awdbWebService"
HEAD = {"Content-Type": "text/xml", "SOAPAction": '""'}

def soap(body_inner):
    env = (f'<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" '
           f'xmlns:web="{NS}"><soapenv:Body>{body_inner}</soapenv:Body></soapenv:Envelope>')
    r = requests.post(URL, data=env.encode(), headers=HEAD, timeout=90)
    r.raise_for_status()
    return r.text

def get_stations(state, network):
    body = (f"<web:getStations><stateCds>{state}</stateCds>"
            f"<networkCds>{network}</networkCds><logicalAnd>true</logicalAnd></web:getStations>")
    txt = soap(body)
    return re.findall(r"<return>([^<]+)</return>", txt)

def get_metadata(triplets):
    items = "".join(f"<stationTriplets>{t}</stationTriplets>" for t in triplets)
    body = f"<web:getStationMetadataMultiple>{items}</web:getStationMetadataMultiple>"
    txt = soap(body)
    out = {}
    # crude parse: each <return> block holds stationTriplet, name, latitude, longitude, elevation
    for block in re.findall(r"<return>(.*?)</return>", txt, re.S):
        def g(tag):
            m = re.search(rf"<{tag}>([^<]*)</{tag}>", block)
            return m.group(1) if m else ""
        trip = g("stationTriplet")
        if trip:
            out[trip] = dict(name=g("name"), lat=g("latitude"), lon=g("longitude"),
                             elev=g("elevation"), county=g("countyName"))
    return out

def show(label, state, network, filt):
    trips = get_stations(state, network)
    meta = get_metadata(trips)
    print(f"\n### {label}: {len(trips)} stations")
    rows = []
    for t, m in meta.items():
        try:
            lat, lon, elev = float(m["lat"]), float(m["lon"]), float(m["elev"])
        except Exception:
            continue
        if filt(lat, lon, elev):
            rows.append((t, m["name"], lat, lon, elev))
    for r in sorted(rows, key=lambda x: -x[4]):
        print(f"  {r[0]:16s} {r[1][:38]:38s} lat={r[2]:.3f} lon={r[3]:.3f} elev_ft={r[4]:.0f}")
    return rows

if __name__ == "__main__":
    # Sierra Nevada SNOTEL for NP15 (Sacramento basin headwaters)
    show("CA SNTL Sierra", "CA", "SNTL", lambda la, lo, el: 36.8 <= la <= 39.8 and -120.9 <= lo <= -118.9 and el >= 5500)
    # Cascades SNOTEL for Mid-Columbia
    show("WA SNTL Cascades", "WA", "SNTL", lambda la, lo, el: 45.5 <= la <= 48.5 and -123.0 <= lo <= -120.5 and el >= 2500)
    show("OR SNTL Cascades", "OR", "SNTL", lambda la, lo, el: 43.0 <= la <= 46.5 and -123.0 <= lo <= -120.8 and el >= 3000)
    # SCAN picks
    show("CA SCAN valley", "CA", "SCAN", lambda la, lo, el: 34.5 <= la <= 40.0 and -122.5 <= lo <= -117.5)
    show("TX SCAN", "TX", "SCAN", lambda la, lo, el: True)
    show("WA SCAN", "WA", "SCAN", lambda la, lo, el: True)
    show("OR SCAN", "OR", "SCAN", lambda la, lo, el: True)
