"use client"

import React, { useState } from "react"
import dynamic from "next/dynamic"
import { X, MapPin, Search, Check, Navigation, Loader2 } from "lucide-react"

const LeafletMap = dynamic(() => import("./LeafletMap"), { ssr: false })

export default function MapAoiPickerModal({ isOpen, onClose, onSelectAoi }) {
  const [searchQuery, setSearchQuery] = useState("")
  const [searching, setSearching] = useState(false)
  const [selectedPoint, setSelectedPoint] = useState(null)
  const [currentCenter, setCurrentCenter] = useState([26.85, 80.95]) // Default Lucknow area
  const [currentBbox, setCurrentBbox] = useState([80.85, 26.75, 81.05, 26.95])
  const [placeLabel, setPlaceLabel] = useState("Lucknow, Uttar Pradesh")
  const [searchError, setSearchError] = useState("")

  if (!isOpen) return null

  async function handleGeocode(e) {
    e?.preventDefault()
    if (!searchQuery.trim()) return

    setSearching(true)
    setSearchError("")

    try {
      // Check if user typed coordinates directly: "26.85, 80.95" or "26.85N, 80.95E"
      const coordMatch = searchQuery.match(/(-?\d+(?:\.\d+)?)\s*°?\s*([NS])?[\s,;]+(-?\d+(?:\.\d+)?)\s*°?\s*([EW])?/i)
      if (coordMatch) {
        let lat = parseFloat(coordMatch[1])
        if (coordMatch[2] && coordMatch[2].toUpperCase() === "S") lat = -lat
        let lon = parseFloat(coordMatch[3])
        if (coordMatch[4] && coordMatch[4].toUpperCase() === "W") lon = -lon

        const delta = 0.08
        const bbox = [
          parseFloat((lon - delta).toFixed(4)),
          parseFloat((lat - delta).toFixed(4)),
          parseFloat((lon + delta).toFixed(4)),
          parseFloat((lat + delta).toFixed(4)),
        ]
        setCurrentCenter([lat, lon])
        setCurrentBbox(bbox)
        setSelectedPoint({ lat, lon, bbox })
        setPlaceLabel(`${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`)
        setSearching(false)
        return
      }

      // Live OpenStreetMap Nominatim Geocoding API
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(searchQuery)}&format=json&limit=1`,
        { headers: { "User-Agent": "SatQueryAI/1.0" } }
      )
      const data = await res.json()
      if (data && data.length > 0) {
        const item = data[0]
        const lat = parseFloat(item.lat)
        const lon = parseFloat(item.lon)
        const bb = item.boundingbox // [south, north, west, east]
        const bbox = [parseFloat(bb[2]), parseFloat(bb[0]), parseFloat(bb[3]), parseFloat(bb[1])]

        setCurrentCenter([lat, lon])
        setCurrentBbox(bbox)
        setSelectedPoint({ lat, lon, bbox })
        setPlaceLabel(item.display_name.split(",").slice(0, 3).join(","))
      } else {
        setSearchError("Location not found. Try searching another landmark or click directly on the map.")
      }
    } catch (err) {
      setSearchError("Geocoding lookup failed. Click anywhere on the map to set the coordinates manually.")
    } finally {
      setSearching(false)
    }
  }

  function handleMapClickSelect({ lat, lon, bbox }) {
    setSelectedPoint({ lat, lon, bbox })
    setCurrentCenter([lat, lon])
    setCurrentBbox(bbox)
    setPlaceLabel(`${lat}°N, ${lon}°E`)
  }

  function handleConfirm() {
    if (selectedPoint) {
      onSelectAoi(selectedPoint)
      onClose()
    } else {
      onSelectAoi({
        lat: currentCenter[0],
        lon: currentCenter[1],
        bbox: currentBbox,
        name: placeLabel,
      })
      onClose()
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 animate-in fade-in">
      <div className="relative flex max-h-[92vh] w-full max-w-4xl flex-col rounded-3xl border border-border bg-card shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-border px-6 py-4 bg-muted/20">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-sky-500/10 text-sky-500">
              <MapPin className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-foreground">Interactive Satellite AOI Selector</h2>
              <p className="text-xs text-muted-foreground">
                Click anywhere on Earth or search a location to define target coordinates for satellite observation
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

        {/* Search Bar */}
        <div className="border-b border-border bg-muted/10 p-4">
          <form onSubmit={handleGeocode} className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search city, district, or coordinates (e.g. 'Lucknow' or '26.85°N, 80.95°E')..."
                className="w-full rounded-xl border border-border bg-background pl-9 pr-4 py-2 text-xs text-foreground placeholder:text-muted-foreground outline-none focus:border-sky-500"
              />
            </div>
            <button
              type="submit"
              disabled={searching || !searchQuery.trim()}
              className="flex items-center gap-1.5 rounded-xl bg-sky-500 px-4 py-2 text-xs font-semibold text-zinc-950 hover:bg-sky-400 disabled:opacity-50 transition-colors"
            >
              {searching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
              <span>Find</span>
            </button>
          </form>
          {searchError && <p className="text-xs text-rose-500 mt-2 font-medium">{searchError}</p>}
        </div>

        {/* Map Body */}
        <div className="relative flex-1 min-h-[380px] p-4 bg-muted/5">
          <LeafletMap
            center={currentCenter}
            bbox={currentBbox}
            placeName={placeLabel}
            allowSelection={true}
            onSelect={handleMapClickSelect}
            height="400px"
          />
        </div>

        {/* Modal Footer */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border bg-muted/30 px-6 py-4">
          <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
            <Navigation className="h-3.5 w-3.5 text-sky-500" />
            <span className="font-semibold text-foreground">Target AOI:</span>
            <span className="bg-muted px-2 py-0.5 rounded text-foreground">
              {placeLabel} [{currentBbox.map((n) => Number(n).toFixed(3)).join(", ")}]
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl border border-border px-4 py-2 text-xs font-medium text-muted-foreground hover:bg-muted transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleConfirm}
              className="flex items-center gap-1.5 rounded-xl bg-sky-500 px-5 py-2 text-xs font-semibold text-zinc-950 hover:bg-sky-400 shadow-md transition-colors"
            >
              <Check className="h-4 w-4" />
              <span>Use This Location in Query</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
