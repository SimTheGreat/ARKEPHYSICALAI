import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import "./App.css";
import { PHASES } from "./constants/phases";
import LiveFeedTile from "./components/LiveFeedTile";
import DefectTile from "./components/DefectTile";
import PhaseFlowTile from "./components/PhaseFlowTile";
import Modal from "./components/Modal";
import ZoneEditorSettings from "./components/settings/ZoneEditorSettings";
import LiveTrackerSettings from "./components/settings/LiveTrackerSettings";
import DefectCameraSettings from "./components/settings/DefectCameraSettings";

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function App() {
  const [error, setError] = useState("");
  const [liveSettings, setLiveSettings] = useState(null);
  const [liveStatus, setLiveStatus] = useState(null);
  const [zones, setZones] = useState([]);
  const [selectedZone, setSelectedZone] = useState("ALL");
  const [editingStation, setEditingStation] = useState(PHASES[0]);
  const [draftPoints, setDraftPoints] = useState([]);
  const [defectSettings, setDefectSettings] = useState(null);
  const [frameTick, setFrameTick] = useState(Date.now());
  const [savingLive, setSavingLive] = useState(false);
  const [savingDefect, setSavingDefect] = useState(false);
  const [savingZones, setSavingZones] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [showZoneModal, setShowZoneModal] = useState(false);
  const [showLiveSettingsModal, setShowLiveSettingsModal] = useState(false);
  const [showDefectSettingsModal, setShowDefectSettingsModal] = useState(false);

  useEffect(() => {
    fetchAll();
    const statusInterval = setInterval(fetchLiveStatus, 1000);
    const frameInterval = setInterval(() => setFrameTick(Date.now()), 350);
    return () => {
      clearInterval(statusInterval);
      clearInterval(frameInterval);
    };
  }, []);

  const currentPhase = liveStatus?.current_station || "NOT_PRESENT";
  const defectEnabled = currentPhase === "Test";
  const phaseAnchor = liveStatus?.completed_phases?.length
    ? liveStatus.completed_phases[liveStatus.completed_phases.length - 1]
    : null;
  const flowPhaseIndex = phaseAnchor ? PHASES.indexOf(phaseAnchor) : -1;
  const frameWidth = Number(liveSettings?.width || 960);
  const frameHeight = Number(liveSettings?.height || 540);

  const visibleZones = useMemo(() => {
    if (selectedZone === "ALL") return zones;
    return zones.filter((zone) => zone.station === selectedZone);
  }, [selectedZone, zones]);

  async function fetchAll() {
    await Promise.all([fetchLiveSettings(), fetchLiveStatus(), fetchZones(), fetchDefectStatus()]);
  }

  async function fetchLiveSettings() {
    try {
      const response = await axios.get("/api/vision/live/settings");
      setLiveSettings(response.data);
      setError("");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to load live settings");
    }
  }

  async function fetchLiveStatus() {
    try {
      const response = await axios.get("/api/vision/live/status");
      setLiveStatus(response.data);
      setError("");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to load live status");
    }
  }

  async function fetchZones() {
    try {
      const response = await axios.get("/api/vision/live/zones");
      setZones(response.data?.polygons || []);
      setError("");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to load zone config");
    }
  }

  async function fetchDefectStatus() {
    try {
      const response = await axios.get("/api/vision/defect/status");
      setDefectSettings(response.data);
      setError("");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to load defect settings");
    }
  }

  function onLiveFieldChange(field, value) {
    setLiveSettings((prev) => ({ ...prev, [field]: value }));
  }

  async function saveLiveSettings() {
    if (!liveSettings) return;
    setSavingLive(true);
    try {
      await axios.post("/api/vision/live/settings", {
        source: String(liveSettings.source),
        width: Number(liveSettings.width),
        height: Number(liveSettings.height),
        min_area: Number(liveSettings.min_area),
        score_decay: Number(liveSettings.score_decay),
        switch_score: Number(liveSettings.switch_score),
        debounce_frames: Number(liveSettings.debounce_frames),
        lower_h: Number(liveSettings.lower_h),
        lower_s: Number(liveSettings.lower_s),
        lower_v: Number(liveSettings.lower_v),
        upper_h: Number(liveSettings.upper_h),
        upper_s: Number(liveSettings.upper_s),
        upper_v: Number(liveSettings.upper_v),
      });
      await fetchLiveSettings();
      setShowLiveSettingsModal(false);
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to save live settings");
    } finally {
      setSavingLive(false);
    }
  }

  function onDefectFieldChange(field, value) {
    setDefectSettings((prev) => ({ ...prev, [field]: value }));
  }

  async function saveDefectSettings() {
    if (!defectSettings) return;
    setSavingDefect(true);
    try {
      await axios.post("/api/vision/defect/settings", {
        source: String(defectSettings.source),
        width: Number(defectSettings.width),
        height: Number(defectSettings.height),
        reference_image_path: defectSettings.reference_image_path,
      });
      await fetchDefectStatus();
      setShowDefectSettingsModal(false);
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to save defect settings");
    } finally {
      setSavingDefect(false);
    }
  }

  function onLiveFrameClick(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const xNorm = clamp01((event.clientX - rect.left) / rect.width);
    const yNorm = clamp01((event.clientY - rect.top) / rect.height);
    setDraftPoints((prev) => [...prev, [xNorm, yNorm]]);
  }

  function loadStationPolygonToDraft() {
    const found = zones.find((zone) => zone.station === editingStation);
    setDraftPoints(found ? found.points_norm : []);
  }

  function undoDraftPoint() {
    setDraftPoints((prev) => prev.slice(0, -1));
  }

  function clearDraft() {
    setDraftPoints([]);
  }

  async function saveDraftZone() {
    if (draftPoints.length < 3) {
      setError("Need at least 3 points for a polygon.");
      return;
    }
    setSavingZones(true);
    try {
      const otherZones = zones.filter((zone) => zone.station !== editingStation);
      const updatedZones = [...otherZones, { station: editingStation, points_norm: draftPoints }];
      await axios.post("/api/vision/live/zones", { polygons: updatedZones });
      setZones(updatedZones);
      setSelectedZone(editingStation);
      setShowZoneModal(false);
      setError("");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to save station zones");
    } finally {
      setSavingZones(false);
    }
  }

  async function deleteStationZone() {
    setSavingZones(true);
    try {
      const updatedZones = zones.filter((zone) => zone.station !== editingStation);
      await axios.post("/api/vision/live/zones", { polygons: updatedZones });
      setZones(updatedZones);
      setDraftPoints([]);
      setError("");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Failed to delete station zone");
    } finally {
      setSavingZones(false);
    }
  }

  async function autoCalibrateForEditingStation() {
    setCalibrating(true);
    try {
      await axios.post("/api/vision/live/auto-calibrate", {
        station: editingStation,
        samples: 10,
      });
      await Promise.all([fetchLiveSettings(), fetchLiveStatus()]);
      setError("");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Auto-calibration failed");
    } finally {
      setCalibrating(false);
    }
  }

  return (
    <div className="dashboard-shell">
      <header className="dashboard-header">
        <div>
          <h1>Physical Production Dashboard</h1>
          <p>Live line tracking, defect camera, and phase flow state</p>
        </div>
        <button onClick={fetchAll}>Refresh Data</button>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="tiles-grid">
        <LiveFeedTile
          currentPhase={currentPhase}
          frameTick={frameTick}
          liveStatus={liveStatus}
          frameWidth={frameWidth}
          frameHeight={frameHeight}
          visibleZones={visibleZones}
          draftPoints={draftPoints}
          onLiveFrameClick={onLiveFrameClick}
          onOpenZoneModal={() => setShowZoneModal(true)}
          onOpenLiveSettingsModal={() => setShowLiveSettingsModal(true)}
        />

        <DefectTile
          frameTick={frameTick}
          onOpenDefectSettingsModal={() => setShowDefectSettingsModal(true)}
          defectEnabled={defectEnabled}
          currentPhase={currentPhase}
        />
      </section>

      <PhaseFlowTile
        phases={PHASES}
        currentPhase={currentPhase}
        flowPhaseIndex={flowPhaseIndex}
      />

      <Modal
        isOpen={showZoneModal}
        title="Zone Settings"
        onClose={() => setShowZoneModal(false)}
      >
        <ZoneEditorSettings
          phases={PHASES}
          editingStation={editingStation}
          setEditingStation={setEditingStation}
          selectedZone={selectedZone}
          setSelectedZone={setSelectedZone}
          visibleZones={visibleZones}
          draftPoints={draftPoints}
          savingZones={savingZones}
          onLoadStation={loadStationPolygonToDraft}
          onUndo={undoDraftPoint}
          onClear={clearDraft}
          onSave={saveDraftZone}
          onDelete={deleteStationZone}
          onAutoCalibrate={autoCalibrateForEditingStation}
          calibrating={calibrating}
        />
      </Modal>

      <Modal
        isOpen={showLiveSettingsModal}
        title="Live Tracker Settings"
        onClose={() => setShowLiveSettingsModal(false)}
      >
        <LiveTrackerSettings
          liveSettings={liveSettings}
          onFieldChange={onLiveFieldChange}
          onSave={saveLiveSettings}
          saving={savingLive}
        />
      </Modal>

      <Modal
        isOpen={showDefectSettingsModal}
        title="Defect Camera Settings"
        onClose={() => setShowDefectSettingsModal(false)}
      >
        <DefectCameraSettings
          defectSettings={defectSettings}
          onFieldChange={onDefectFieldChange}
          onSave={saveDefectSettings}
          saving={savingDefect}
        />
      </Modal>
    </div>
  );
}

export default App;
