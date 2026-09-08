"use client"

import React, { useState, useEffect } from "react"
import dynamic from "next/dynamic"
import { X, Search, Satellite, Calendar, MapPin, CheckCircle2, Download, AlertCircle, Loader2, Navigation, Layers } from "lucide-react"
import { searchBhoonidhi } from "@/lib/satquery"

const LeafletMap = dynamic(() => import("./LeafletMap"), { ssr: false })

const COMMON_SATELLITES = [
  { name: "Sentinel-2A", sensor: "MSI", desc: "10m Optical MSI (OpenData)" },
  { name: "EOS-04", sensor: "SAR(MRS)", desc: "C-band Microwave SAR Radar" },
  { name: "CartoSat-2S", sensor: "MX(SPOT)", desc: "Sub-metre High-Res Optical" },
  { name: "CartoSat-3", sensor: "PAN(SPOT)", desc: "0.28m Highest-Res Panchromatic" },
  { name: "ResourceSat-2", sensor: "LISS4(MX23)", desc: "5.8m Multispectral" },
]

export default function BhoonidhiSearchModal({ isOpen, onClose, onSelectScene }) {
  const [satellite, setSatellite] = useState("Sentinel-2A")
  const [sensor, setSensor] = useState("MSI")
  const [locationName, setLocationName] = useState("Lucknow, Uttar Pradesh")
  const [bbox, setBbox] = useState([80.85, 26.75, 81.05, 26.95])
  const [center, setCenter] = useState([26.85, 80.95])
  const [showMap, setShowMap] = useState(false)

  // Dynamic dates computed relative to runtime
  const [startDate, setStartDate] = useState(() => {
    const d = new Date()
    d.setMonth(d.getMonth() - 6)
    return d.toISOString().slice(0, 10)
  })
  const [endDate, setEndDate] = useState(() => new Date().toISOString().slice(0, 10))

  const [loading, setLoading] = useState(false)
  const [geocoding, setGeocoding] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  function handleSatelliteChange(satName) {
    setSatellite(satName)
    const match = COMMON_SATELLITES.find((s) => s.name === satName)
    if (match) setSensor(match.sensor)
  }

  async function handleGeocodeLocation(e) {
    e?.preventDefault()
    if (!locationName.trim()) return

    setGeocoding(true)
    setError(null)

    try {
      // Check for coordinate format
      const coordMatch = locationName.match(/(-?\d+(?:\.\d+)?)\s*°?\s*([NS])?[\s,;]+(-?\d+(?:\.\d+)?)\s*°?\s*([EW])?/i)
      if (coordMatch) {
        let lat = parseFloat(coordMatch[1])
        if (coordMatch[2] && coordMatch[2].toUpperCase() === "S") lat = -lat
        let lon = parseFloat(coordMatch[3])
        if (coordMatch[4] && coordMatch[4].toUpperCase() === "W") lon = -lon

        const delta = 0.1
        const newBbox = [
          parseFloat((lon - delta).toFixed(4)),
          parseFloat((lat - delta).toFixed(4)),
          parseFloat((lon + delta).toFixed(4)),
          parseFloat((lat + delta).toFixed(4)),
        ]
        setCenter([lat, lon])
        setBbox(newBbox)
        setGeocoding(false)
        return
      }

      // Live OpenStreetMap Nominatim Geocoding
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(locationName)}&format=json&limit=1`,
        { headers: { "User-Agent": "SatQueryAI/1.0" } }
      )
      const data = await res.json()
      if (data && data.length > 0) {
        const item = data[0]
        const lat = parseFloat(item.lat)
        const lon = parseFloat(item.lon)
        const bb = item.boundingbox
        const newBbox = [parseFloat(bb[2]), parseFloat(bb[0]), parseFloat(bb[3]), parseFloat(bb[1])]

        setCenter([lat, lon])
        setBbox(newBbox)
        setLocationName(item.display_name.split(",").slice(0, 3).join(","))
      } else {
        setError("Could not resolve location. Try searching another place or use coordinates.")
      }
    } catch (err) {
      setError("Geocoding failed. You can adjust the coordinates directly.")
    } finally {
      setGeocoding(false)
    }
  }

  function handleMapSelect({ lat, lon, bbox: newBbox }) {
    setCenter([lat, lon])
    setBbox(newBbox)
    setLocationName(`${lat}°N, ${lon}°E`)
  }

  async function handleSearch(e) {
    e?.preventDefault()
    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const res = await searchBhoonidhi({
        satellite,
        sensor: sensor || undefined,
        start_date: startDate,
        end_date: endDate,
        bbox,
        limit: 12,
      })
      if (res.error) {
        setError(res.error)
      } else {
        setResults(res)
      }
    } catch (err) {
      setError(err.message || "Failed to query ISRO Bhoonidhi catalog")
    } finally {
      setLoading(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 animate-in fade-in">
      <div className="relative flex max-h-[92vh] w-full max-w-3xl flex-col rounded-3xl border border-border bg-card shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-border px-6 py-4 bg-muted/20">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-sky-500/10 text-sky-500">
              <Satellite className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-foreground">ISRO Bhoonidhi STAC Explorer</h2>
              <p className="text-xs text-muted-foreground">
                Query dynamic Indian Remote Sensing scenes from NRSC Bhoonidhi portal
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-full p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Search Controls */}
        <div className="border-b border-border bg-muted/30 p-5">
          <form onSubmit={handleSearch} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* Satellite Selector */}
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">Satellite Mission</label>
                <select
                  value={satellite}
                  onChange={(e) => handleSatelliteChange(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-3 py-2 text-xs font-medium text-foreground outline-none focus:ring-1 focus:ring-sky-500"
                >
                  {COMMON_SATELLITES.map((s) => (
                    <option key={s.name} value={s.name}>
                      {s.name} · {s.desc}
                    </option>
                  ))}
                </select>
              </div>

              {/* Dynamic Location Input with Live Geocoding */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-xs font-medium text-muted-foreground">Target AOI / Location</label>
                  <button
                    type="button"
                    onClick={() => setShowMap(!showMap)}
                    className="text-[11px] text-sky-500 hover:underline flex items-center gap-1"
                  >
                    <MapPin className="h-3 w-3" />
                    <span>{showMap ? "Hide Map" : "Interactive Map"}</span>
                  </button>
                </div>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={locationName}
                    onChange={(e) => setLocationName(e.target.value)}
                    placeholder="Enter city, district or coordinates..."
                    className="flex-1 rounded-xl border border-border bg-background px-3 py-2 text-xs text-foreground outline-none focus:ring-1 focus:ring-sky-500"
                  />
                  <button
                    type="button"
                    onClick={handleGeocodeLocation}
                    disabled={geocoding}
                    className="rounded-xl border border-border bg-background px-3 py-2 text-xs font-medium text-foreground hover:bg-muted"
                  >
                    {geocoding ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Locate"}
                  </button>
                </div>
              </div>
            </div>

            {/* Optional Collapsible Leaflet Map for visual AOI selection */}
            {showMap && (
              <div className="overflow-hidden rounded-2xl border border-border bg-background p-2">
                <LeafletMap
                  center={center}
                  bbox={bbox}
                  placeName={locationName}
                  allowSelection={true}
                  onSelect={handleMapSelect}
                  height="220px"
                />
              </div>
            )}

            {/* Date Range */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-3 py-1.5 text-xs text-foreground outline-none focus:ring-1 focus:ring-sky-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">End Date</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full rounded-xl border border-border bg-background px-3 py-1.5 text-xs text-foreground outline-none focus:ring-1 focus:ring-sky-500"
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-1">
              <span className="text-[11px] font-mono text-muted-foreground truncate max-w-sm">
                BBOX: [{bbox.map((n) => Number(n).toFixed(3)).join(", ")}]
              </span>
              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center gap-1.5 rounded-xl bg-sky-500 px-4 py-2 text-xs font-semibold text-zinc-950 hover:bg-sky-400 disabled:opacity-50 transition-colors shadow-sm"
              >
                {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
                <span>Search Bhoonidhi Catalog</span>
              </button>
            </div>
          </form>
        </div>

        {/* Search Results Display */}
        <div className="flex-1 overflow-y-auto p-5 space-y-3">
          {loading && (
            <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
              <Loader2 className="h-8 w-8 animate-spin text-sky-500 mb-3" />
              <p className="text-sm font-medium text-foreground">Querying ISRO NRSC Bhoonidhi catalog...</p>
              <p className="text-xs text-muted-foreground mt-1">Retrieving pass metadata and footprints</p>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2.5 rounded-2xl border border-rose-500/20 bg-rose-500/10 p-4 text-xs text-rose-500 dark:text-rose-400">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Bhoonidhi Query Notice</p>
                <p className="mt-0.5">{error}</p>
              </div>
            </div>
          )}

          {results && results.scenes && results.scenes.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-center text-muted-foreground">
              <Satellite className="h-10 w-10 text-muted-foreground/40 mb-2" />
              <p className="text-sm font-medium text-foreground">No scenes found for this bounding box and date range</p>
              <p className="text-xs text-muted-foreground mt-1">
                Try widening your date window or searching another Indian region.
              </p>
            </div>
          )}

          {results && results.scenes && results.scenes.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs text-muted-foreground px-1 pb-1">
                <span>{results.count} Satellite Scenes Available</span>
                <span className="font-mono text-[11px]">Direct STAC Stream</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {results.scenes.map((scene) => (
                  <div
                    key={scene.id}
                    className="flex flex-col justify-between rounded-2xl border border-border bg-card p-3.5 hover:border-sky-500/50 hover:bg-muted/20 transition-all shadow-xs"
                  >
                    <div>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="rounded-md bg-sky-500/10 px-2 py-0.5 text-[10px] font-semibold text-sky-500">
                          {scene.satellite}
                        </span>
                        <span className="text-[11px] font-mono text-muted-foreground">
                          {scene.dop || "Recent"}
                        </span>
                      </div>
                      <p className="text-xs font-semibold text-foreground truncate" title={scene.id}>
                        {scene.id}
                      </p>
                      <p className="text-[11px] text-muted-foreground mt-0.5">
                        Sensor: {scene.sensor || "MSI"} · Res: {scene.resolution || "10m"}
                      </p>
                    </div>

                    <div className="mt-3 flex items-center justify-between pt-2 border-t border-border/60">
                      <span className="text-[10px] font-mono text-emerald-500 font-medium">
                        {scene.access_type || "OpenData"}
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          onSelectScene?.(scene)
                          onClose()
                        }}
                        className="inline-flex items-center gap-1 rounded-xl bg-sky-500 px-3 py-1 text-[11px] font-semibold text-zinc-950 hover:bg-sky-400 transition-colors"
                      >
                        <CheckCircle2 className="h-3 w-3" />
                        <span>Select Scene</span>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
