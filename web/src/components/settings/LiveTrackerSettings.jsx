function LiveTrackerSettings({ liveSettings, onFieldChange, onSave, saving }) {
  if (!liveSettings) {
    return <p>Loading settings...</p>;
  }

  return (
    <div className="tool-panel">
      <h3>HSV + Tracker Settings</h3>
      <div className="field-grid">
        <label>
          Source
          <input
            value={liveSettings.source}
            onChange={(e) => onFieldChange("source", e.target.value)}
          />
        </label>
        <label>
          Min Area
          <input
            type="number"
            value={liveSettings.min_area}
            onChange={(e) => onFieldChange("min_area", e.target.value)}
          />
        </label>
        <label>
          Width
          <input
            type="number"
            value={liveSettings.width}
            onChange={(e) => onFieldChange("width", e.target.value)}
          />
        </label>
        <label>
          Height
          <input
            type="number"
            value={liveSettings.height}
            onChange={(e) => onFieldChange("height", e.target.value)}
          />
        </label>
        <label>
          Decay
          <input
            type="number"
            step="0.01"
            value={liveSettings.score_decay}
            onChange={(e) => onFieldChange("score_decay", e.target.value)}
          />
        </label>
        <label>
          Switch Score
          <input
            type="number"
            step="0.01"
            value={liveSettings.switch_score}
            onChange={(e) => onFieldChange("switch_score", e.target.value)}
          />
        </label>
        <label>
          Debounce
          <input
            type="number"
            value={liveSettings.debounce_frames}
            onChange={(e) => onFieldChange("debounce_frames", e.target.value)}
          />
        </label>
      </div>
      <div className="slider-grid">
        {[
          ["lower_h", 0, 179],
          ["lower_s", 0, 255],
          ["lower_v", 0, 255],
          ["upper_h", 0, 179],
          ["upper_s", 0, 255],
          ["upper_v", 0, 255],
        ].map(([key, min, max]) => (
          <label key={key}>
            {key}: {liveSettings[key]}
            <input
              type="range"
              min={min}
              max={max}
              value={liveSettings[key]}
              onChange={(e) => onFieldChange(key, Number(e.target.value))}
            />
          </label>
        ))}
      </div>
      <button onClick={onSave} disabled={saving}>
        {saving ? "Saving..." : "Save Live Settings"}
      </button>
    </div>
  );
}

export default LiveTrackerSettings;
