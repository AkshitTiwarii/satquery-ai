"use client"

import React, { useEffect, useRef, useState } from "react"
import { Layers, MapPin, Maximize2, Minimize2, Crosshair, Navigation, Info } from "lucide-react"

export default function LeafletMap({
  bbox = null,
  center = null,
  zoom = 11,
  geojson = null,
  placeName = "",
  allowSelection = false,
  onSelect = null,
  height = "360px",
  className = "",
  showControls = true,
}) {
  const mapContainerRef = useRef(null)
  const mapInstanceRef = useRef(null)
  const layersRef = useRef({
    tileLayer: null,
    bboxLayer: null,
    geojsonLayer: null,
    selectionMarker: null,
  })

  const [currentBase, setCurrentBase] = useState("satellite") // "satellite" | "dark" | "osm"
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [coords, setCoords] = useState(null)
  const [selectedCoords, setSelectedCoords] = useState(center ? { lat: center[0], lon: center[1] } : null)

  // Initialize Leaflet map client-side
  useEffect(() => {
    let isMounted = true
    let map = null

    async function initMap() {
      if (!mapContainerRef.current) return

      // Dynamically import Leaflet to avoid SSR window errors
      const L = (await import("leaflet")).default

      // Fix default Leaflet icon paths
      delete L.Icon.Default.prototype._getIconUrl
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
        iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
        shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      })

      // Determine initial center
      let initialCenter = [20.5937, 78.9629] // Center of India default
      let initialZoom = 5

      if (center && center.length === 2) {
        initialCenter = [Number(center[0]), Number(center[1])]
        initialZoom = zoom || 11
      } else if (bbox && bbox.length === 4) {
        // [minx, miny, maxx, maxy] -> [minLon, minLat, maxLon, maxLat]
        const latCenter = (Number(bbox[1]) + Number(bbox[3])) / 2
        const lonCenter = (Number(bbox[0]) + Number(bbox[2])) / 2
        initialCenter = [latCenter, lonCenter]
        initialZoom = zoom || 11
      }

      // Cleanup existing map if any
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove()
        mapInstanceRef.current = null
      }

      // Create map
      map = L.map(mapContainerRef.current, {
        center: initialCenter,
        zoom: initialZoom,
        zoomControl: false,
        attributionControl: false,
      })

      mapInstanceRef.current = map

      // Base tile layers
      const baseTileUrls = {
        satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        dark: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        osm: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      }

      const activeUrl = baseTileUrls[currentBase] || baseTileUrls.satellite
      const tileLayer = L.tileLayer(activeUrl, {
        maxZoom: 19,
        attribution: "Esri / ISRO / OpenStreetMap",
      }).addTo(map)

      layersRef.current.tileLayer = tileLayer

      // Mouse coordinate tracker
      map.on("mousemove", (e) => {
        if (isMounted) {
          setCoords({
            lat: e.latlng.lat.toFixed(4),
            lng: e.latlng.lng.toFixed(4),
          })
        }
      })

      // Click to select AOI
      if (allowSelection) {
        map.on("click", (e) => {
          const lat = parseFloat(e.latlng.lat.toFixed(4))
          const lon = parseFloat(e.latlng.lng.toFixed(4))
          const delta = 0.08 // ~8km bounding box
          const generatedBbox = [
            parseFloat((lon - delta).toFixed(4)),
            parseFloat((lat - delta).toFixed(4)),
            parseFloat((lon + delta).toFixed(4)),
            parseFloat((lat + delta).toFixed(4)),
          ]

          setSelectedCoords({ lat, lon })

          // Update selection marker
          if (layersRef.current.selectionMarker) {
            layersRef.current.selectionMarker.remove()
          }

          const pulseIcon = L.divIcon({
            className: "custom-pulse-marker",
            html: `<div style="
              width: 20px;
              height: 20px;
              background: rgba(14, 165, 233, 0.4);
              border: 2px solid #38bdf8;
              border-radius: 50%;
              box-shadow: 0 0 15px #38bdf8;
              position: relative;
            "><div style="
              width: 8px;
              height: 8px;
              background: #38bdf8;
              border-radius: 50%;
              position: absolute;
              top: 4px;
              left: 4px;
            "></div></div>`,
            iconSize: [20, 20],
            iconAnchor: [10, 10],
          })

          layersRef.current.selectionMarker = L.marker([lat, lon], { icon: pulseIcon }).addTo(map)

          if (onSelect) {
            onSelect({ lat, lon, bbox: generatedBbox })
          }
        })
      }

      // Draw BBOX layer if provided
      if (bbox && bbox.length === 4) {
        const minLon = Number(bbox[0])
        const minLat = Number(bbox[1])
        const maxLon = Number(bbox[2])
        const maxLat = Number(bbox[3])

        const bounds = [
          [minLat, minLon],
          [maxLat, maxLon],
        ]

        const bboxRect = L.rectangle(bounds, {
          color: "#38bdf8",
          weight: 2,
          fillColor: "#0284c7",
          fillOpacity: 0.15,
          dashArray: "4, 4",
        }).addTo(map)

        layersRef.current.bboxLayer = bboxRect
        try {
          map.fitBounds(bounds, { padding: [30, 30], animate: false })
        } catch (e) {}
      }

      // Draw Center Pin if given and no bbox
      if (center && (!bbox || bbox.length !== 4)) {
        const pin = L.circleMarker([center[0], center[1]], {
          radius: 8,
          fillColor: "#38bdf8",
          color: "#ffffff",
          weight: 2,
          opacity: 1,
          fillOpacity: 0.8,
        }).addTo(map)
        pin.bindPopup(`<strong>Target Location:</strong><br/>${placeName || `${center[0]}°N, ${center[1]}°E`}`)
      }

      // Render GeoJSON overlay if provided
      if (geojson) {
        try {
          const parsedGeo = typeof geojson === "string" ? JSON.parse(geojson) : geojson
          const geoLayer = L.geoJSON(parsedGeo, {
            style: (feature) => {
              const props = feature.properties || {}
              const label = (props.class || props.label || props.name || "").toLowerCase()

              if (label.includes("water")) {
                return { color: "#06b6d4", weight: 2, fillColor: "#0891b2", fillOpacity: 0.45 }
              }
              if (label.includes("built") || label.includes("urban") || label.includes("building")) {
                return { color: "#f59e0b", weight: 2, fillColor: "#d97706", fillOpacity: 0.45 }
              }
              if (label.includes("change") || label.includes("increase") || label.includes("cluster")) {
                return { color: "#f43f5e", weight: 2.5, fillColor: "#e11d48", fillOpacity: 0.55 }
              }
              return { color: "#38bdf8", weight: 2, fillColor: "#0284c7", fillOpacity: 0.3 }
            },
            onEachFeature: (feature, layer) => {
              const p = feature.properties || {}
              const title = p.label || p.class || p.name || "Remote Sensing Feature"
              const conf = p.confidence ? `${(p.confidence * 100).toFixed(1)}%` : "Verified"
              const area = p.area_ha ? `${p.area_ha} ha` : p.area_km2 ? `${p.area_km2} km²` : "Detected Extent"

              layer.bindPopup(`
                <div style="font-family: monospace; font-size: 11px; padding: 4px; color: #1e293b;">
                  <strong style="color: #0f172a; font-size: 12px;">${title}</strong><br/>
                  <span>Area: ${area}</span><br/>
                  <span>Confidence: ${conf}</span>
                </div>
              `)
            },
          }).addTo(map)

          layersRef.current.geojsonLayer = geoLayer

          // Fit to geojson bounds if no explicit bbox was fitted
          if (!bbox && geoLayer.getBounds().isValid()) {
            try {
              map.fitBounds(geoLayer.getBounds(), { padding: [30, 30], animate: false })
            } catch (e) {}
          }
        } catch (err) {
          console.error("LeafletMap: Failed to parse geojson overlay", err)
        }
      }
    }

    initMap()

    return () => {
      isMounted = false
      if (mapInstanceRef.current) {
        try {
          mapInstanceRef.current.stop()
          mapInstanceRef.current.remove()
        } catch (e) {}
        mapInstanceRef.current = null
      }
    }
  }, [bbox, center, geojson, allowSelection])

  // Handle tile switch cleanly without tearing down the map
  async function switchBaseLayer(type) {
    setCurrentBase(type)
    const map = mapInstanceRef.current
    if (!map) return

    try {
      const L = (await import("leaflet")).default

      if (layersRef.current.tileLayer) {
        try {
          map.removeLayer(layersRef.current.tileLayer)
        } catch (e) {}
      }

      const urls = {
        satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        dark: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        osm: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      }

      const newLayer = L.tileLayer(urls[type] || urls.satellite, {
        maxZoom: 19,
        attribution: "Esri / ISRO / OpenStreetMap",
      }).addTo(map)

      layersRef.current.tileLayer = newLayer
    } catch (err) {
      console.warn("LeafletMap: Failed to switch base tile layer", err)
    }
  }

  // Recenter map safely without animation crash
  function handleRecenter() {
    const map = mapInstanceRef.current
    if (!map) return
    try {
      if (layersRef.current.bboxLayer && layersRef.current.bboxLayer.getBounds().isValid()) {
        map.fitBounds(layersRef.current.bboxLayer.getBounds(), { padding: [30, 30], animate: false })
      } else if (center) {
        map.setView([center[0], center[1]], zoom || 11, { animate: false })
      }
    } catch (e) {}
  }

  // Dynamically compute legend items from features actually present in GeoJSON
  const activeLegends = React.useMemo(() => {
    if (!geojson) return []
    try {
      const parsed = typeof geojson === "string" ? JSON.parse(geojson) : geojson
      const features = parsed.features || (parsed.geometry ? [parsed] : [])
      const items = []
      const seen = new Set()

      features.forEach((f) => {
        const props = f.properties || {}
        const name = (props.name || props.label || props.category || props.class || "").toLowerCase()
        const color = props.color || props.fillColor

        if ((name.includes("aoi") || props.category === "aoi_boundary") && !seen.has("aoi")) {
          seen.add("aoi")
          items.push({ label: "Target AOI", color: color || "#38bdf8" })
        } else if ((name.includes("built") || name.includes("urban") || name.includes("structure")) && !seen.has("builtup")) {
          seen.add("builtup")
          items.push({ label: "Built-up", color: color || "#f59e0b" })
        } else if ((name.includes("water") || name.includes("river") || name.includes("wetland")) && !seen.has("water")) {
          seen.add("water")
          items.push({ label: "Water", color: color || "#06b6d4" })
        } else if ((name.includes("change") || name.includes("transition")) && !seen.has("change")) {
          seen.add("change")
          items.push({ label: "Change", color: color || "#f43f5e" })
        } else if ((name.includes("grounding") || name.includes("localized")) && !seen.has("grounding")) {
          seen.add("grounding")
          items.push({ label: "Detected Extent", color: color || "#38bdf8" })
        }
      })
      return items
    } catch {
      return []
    }
  }, [geojson])

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-border bg-card shadow-sm transition-all duration-300 ${
        isFullscreen ? "fixed inset-4 z-50 h-[calc(100vh-2rem)]" : ""
      } ${className}`}
      style={{ height: isFullscreen ? "calc(100vh - 2rem)" : height }}
    >
      {/* The Leaflet DOM Node */}
      <div ref={mapContainerRef} className="h-full w-full z-0 cursor-crosshair" />

      {/* Floating HUD Top Bar */}
      {showControls && (
        <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
          <div className="flex items-center gap-2 pointer-events-auto bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-xl border border-white/10 shadow-lg text-white text-xs">
            <Navigation className="h-3.5 w-3.5 text-sky-400 shrink-0" />
            <span className="font-semibold text-zinc-100 truncate max-w-[200px] sm:max-w-xs">
              {placeName || (center ? `${center[0]}°N, ${center[1]}°E` : "Spatial Grounding View")}
            </span>
            {allowSelection && (
              <span className="rounded bg-sky-500/20 px-1.5 py-0.5 text-[10px] text-sky-300 border border-sky-500/30">
                Click map to select
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5 pointer-events-auto">
            {/* Layer Switcher */}
            <div className="flex items-center bg-black/70 backdrop-blur-md rounded-xl border border-white/10 p-0.5 text-xs text-white shadow-lg">
              <button
                type="button"
                onClick={() => switchBaseLayer("satellite")}
                className={`px-2 py-1 rounded-lg transition-colors ${
                  currentBase === "satellite" ? "bg-sky-500 text-black font-semibold" : "hover:text-sky-300"
                }`}
                title="Satellite Imagery"
              >
                Satellite
              </button>
              <button
                type="button"
                onClick={() => switchBaseLayer("dark")}
                className={`px-2 py-1 rounded-lg transition-colors ${
                  currentBase === "dark" ? "bg-sky-500 text-black font-semibold" : "hover:text-sky-300"
                }`}
                title="Dark Vector Base"
              >
                Dark
              </button>
              <button
                type="button"
                onClick={() => switchBaseLayer("osm")}
                className={`px-2 py-1 rounded-lg transition-colors ${
                  currentBase === "osm" ? "bg-sky-500 text-black font-semibold" : "hover:text-sky-300"
                }`}
                title="OpenStreetMap Streets"
              >
                Streets
              </button>
            </div>

            {/* Recenter */}
            <button
              type="button"
              onClick={handleRecenter}
              className="flex h-8 w-8 items-center justify-center rounded-xl bg-black/70 backdrop-blur-md border border-white/10 text-white shadow-lg hover:bg-black/90 hover:text-sky-400 transition-colors"
              title="Recenter"
            >
              <Crosshair className="h-4 w-4" />
            </button>

            {/* Fullscreen Toggle */}
            <button
              type="button"
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="flex h-8 w-8 items-center justify-center rounded-xl bg-black/70 backdrop-blur-md border border-white/10 text-white shadow-lg hover:bg-black/90 hover:text-sky-400 transition-colors"
              title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </button>
          </div>
        </div>
      )}

      {/* Floating HUD Bottom Coordinates Footer */}
      <div className="absolute bottom-2 left-2 right-2 z-10 flex items-center justify-between pointer-events-none text-[10px] font-mono">
        <div className="bg-black/70 backdrop-blur-md px-2.5 py-1 rounded-lg border border-white/10 text-zinc-300 pointer-events-auto flex items-center gap-2 shadow-md">
          <span className="text-sky-400">ISRO WMS/STAC:</span>
          <span>{coords ? `${coords.lat}°N, ${coords.lng}°E` : "10m GSD Tile Matrix"}</span>
        </div>

        {activeLegends.length > 0 && (
          <div className="bg-black/70 backdrop-blur-md px-2.5 py-1 rounded-lg border border-white/10 text-zinc-300 pointer-events-auto flex items-center gap-3 shadow-md">
            {activeLegends.map((item, idx) => (
              <span key={idx} className="flex items-center gap-1">
                <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: item.color }} /> {item.label}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
