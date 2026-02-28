import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import "./App.css";

const NOT_PRESENT = "NOT_PRESENT";

function App() {
  const [lineState, setLineState] = useState({
    stations: [],
    parts: [],
    by_station: {},
    not_present: [],
    recent_events: [],
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [newPartId, setNewPartId] = useState("");
  const [newPartLabel, setNewPartLabel] = useState("");
  const [selectedPartId, setSelectedPartId] = useState("");
  const [selectedStation, setSelectedStation] = useState(NOT_PRESENT);
  const [confidence, setConfidence] = useState("0.95");
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    fetchState();
    const interval = setInterval(fetchState, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!selectedPartId && lineState.parts.length > 0) {
      setSelectedPartId(lineState.parts[0].part_id);
    }
  }, [lineState.parts, selectedPartId]);

  const stationOptions = useMemo(
    () => [...lineState.stations, NOT_PRESENT],
    [lineState.stations]
  );

  async function fetchState() {
    try {
      const response = await axios.get("/api/line/state");
      setLineState(response.data);
      setError("");
    } catch (err) {
      setError(err?.message || "Failed to load state");
    } finally {
      setLoading(false);
    }
  }

  async function createPart(event) {
    event.preventDefault();
    setIsSubmitting(true);
    try {
      const payload = {};
      if (newPartId.trim()) payload.part_id = newPartId.trim();
      if (newPartLabel.trim()) payload.label = newPartLabel.trim();
      await axios.post("/api/line/parts", payload);
      setNewPartId("");
      setNewPartLabel("");
      await fetchState();
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to create part");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function updateDetection(event) {
    event.preventDefault();
    if (!selectedPartId) return;
    setIsSubmitting(true);
    try {
      const parsedConfidence = Number(confidence);
      const payload = {
        station: selectedStation,
        source: "manual_ui",
        confidence: Number.isFinite(parsedConfidence) ? parsedConfidence : null,
      };
      await axios.post(`/api/line/parts/${selectedPartId}/detection`, payload);
      await fetchState();
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to update detection");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function resetState() {
    setIsSubmitting(true);
    try {
      await axios.post("/api/line/reset");
      setSelectedPartId("");
      await fetchState();
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to reset");
    } finally {
      setIsSubmitting(false);
    }
  }

  if (loading) {
    return <div className="screen-center">Loading line state...</div>;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>Production Line Core</h1>
          <p>Station progression state + event log</p>
        </div>
        <div className="topbar-actions">
          <button onClick={fetchState} disabled={isSubmitting}>
            Refresh
          </button>
          <button className="danger" onClick={resetState} disabled={isSubmitting}>
            Reset
          </button>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="controls-grid">
        <form className="panel" onSubmit={createPart}>
          <h2>Create Part</h2>
          <label>
            Part ID (optional)
            <input
              value={newPartId}
              onChange={(e) => setNewPartId(e.target.value)}
              placeholder="PART-001"
            />
          </label>
          <label>
            Label (optional)
            <input
              value={newPartLabel}
              onChange={(e) => setNewPartLabel(e.target.value)}
              placeholder="Blue PCB sample"
            />
          </label>
          <button type="submit" disabled={isSubmitting}>
            Add Part
          </button>
        </form>

        <form className="panel" onSubmit={updateDetection}>
          <h2>Manual Detection Update</h2>
          <label>
            Part
            <select
              value={selectedPartId}
              onChange={(e) => setSelectedPartId(e.target.value)}
            >
              <option value="">Select part</option>
              {lineState.parts.map((part) => (
                <option key={part.part_id} value={part.part_id}>
                  {part.part_id}
                </option>
              ))}
            </select>
          </label>
          <label>
            Station
            <select
              value={selectedStation}
              onChange={(e) => setSelectedStation(e.target.value)}
            >
              {stationOptions.map((station) => (
                <option key={station} value={station}>
                  {station}
                </option>
              ))}
            </select>
          </label>
          <label>
            Confidence
            <input
              value={confidence}
              onChange={(e) => setConfidence(e.target.value)}
              placeholder="0.95"
            />
          </label>
          <button type="submit" disabled={isSubmitting || !selectedPartId}>
            Update Part Station
          </button>
        </form>
      </section>

      <section className="panel">
        <h2>Live Station View</h2>
        <div className="station-grid">
          {lineState.stations.map((station) => (
            <div className="station-card" key={station}>
              <div className="station-header">
                <span>{station}</span>
                <strong>{lineState.by_station?.[station]?.length || 0}</strong>
              </div>
              <div className="station-body">
                {(lineState.by_station?.[station] || []).length === 0 ? (
                  <p className="muted">No parts</p>
                ) : (
                  (lineState.by_station?.[station] || []).map((part) => (
                    <article className="part-chip" key={part.part_id}>
                      <div className="chip-title">{part.part_id}</div>
                      <div className="chip-sub">{part.label || "No label"}</div>
                    </article>
                  ))
                )}
              </div>
            </div>
          ))}
          <div className="station-card not-present">
            <div className="station-header">
              <span>{NOT_PRESENT}</span>
              <strong>{lineState.not_present?.length || 0}</strong>
            </div>
            <div className="station-body">
              {(lineState.not_present || []).length === 0 ? (
                <p className="muted">No parts</p>
              ) : (
                (lineState.not_present || []).map((part) => (
                  <article className="part-chip" key={part.part_id}>
                    <div className="chip-title">{part.part_id}</div>
                    <div className="chip-sub">{part.label || "No label"}</div>
                  </article>
                ))
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Part Progress</h2>
        <div className="progress-list">
          {lineState.parts.length === 0 ? (
            <p className="muted">No parts created yet.</p>
          ) : (
            lineState.parts.map((part) => (
              <div className="progress-item" key={part.part_id}>
                <div className="progress-head">
                  <strong>{part.part_id}</strong>
                  <span>{part.current_station}</span>
                </div>
                <div className="progress-track">
                  {lineState.stations.map((station, index) => (
                    <div
                      key={station}
                      className={[
                        "progress-step",
                        index <= part.progress_index ? "done" : "",
                        station === part.current_station ? "active" : "",
                      ].join(" ")}
                      title={station}
                    >
                      {station}
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="panel">
        <h2>Recent Transitions</h2>
        <div className="events-table">
          <div className="events-row head">
            <span>Time</span>
            <span>Part</span>
            <span>From</span>
            <span>To</span>
            <span>Source</span>
          </div>
          {[...(lineState.recent_events || [])].reverse().map((event) => (
            <div className="events-row" key={event.event_id}>
              <span>{new Date(event.happened_at).toLocaleTimeString()}</span>
              <span>{event.part_id}</span>
              <span>{event.from_station}</span>
              <span>{event.to_station}</span>
              <span>{event.source}</span>
            </div>
          ))}
          {(lineState.recent_events || []).length === 0 ? (
            <div className="events-row">
              <span className="muted">No transitions yet.</span>
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}

export default App;
